import os
import sys
import json
import sqlite3

METHOD = 'face-1.0.1'

try:
    import stashapi.log as log
    import stashapi.marker_parse as mp
    from stashapi.stashapp import StashInterface

    import cv2
    import insightface
    import numpy as np
    from datetime import timedelta
    import onnxruntime as ort
    from tqdm import tqdm

    insightface_version = insightface.__version__

except ModuleNotFoundError:
    print("You need to install the stashapp-tools (stashapi) python module. (CLI: pip install stashapp-tools)", file=sys.stderr)

try:
    log.info(f"Available providers: {ort.get_available_providers()}")
#    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = ['CUDAExecutionProvider']
    model = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    model.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    log.error(f"Error initializing InsightFace model: {e}")
    raise

# plugins don't start in the right directory, let's switch to the local directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def exit_plugin(msg=None, err=None):
    if msg is None and err is None:
        msg = "plugin ended"
    output_json = {"output": msg, "error": err}
    print(json.dumps(output_json))
    sys.exit()

def catchup():
    #f = {"stash_ids": {"modifier": "NOT_NULL"}}

    f = {
            "stash_id_endpoint": {
                "modifier": "NOT_NULL",
            }
        }
#                "stash_id": {"modifier": "NOT_NULL"}
    log.info('Getting scene count.')
    count=stash.find_scenes(f=f,filter={},get_count=True)[0]

    log.info(str(count)+' scenes to extract faces.')
    i=0
    for r in range(1,count+1):
        log.info('fetching data: %s - %s %0.1f%%' % ((r - 1) * 1,r,(i/count)*100,))
#        scenes=stash.find_scenes(f=f,filter={"page":r, "per_page": 1, "sort": "duration", "direction": "ASC"})
        scenes=stash.find_scenes(f=f,filter={"page":r, "per_page": 1, "sort": "title", "direction": "ASC"})
        for s in scenes:
            if "stash_ids" not in s.keys() or len(s["stash_ids"]) != 1:
                log.error(f"Scene {s['id']} must have exactly one stash_id, skipping...")
                continue
            result = checkface(s)
            #processScene(s)
            i=i+1
            log.progress((i/count))
            #time.sleep(2)

def checkface(scene):
    #log.info(scene)

    if len(scene['files']) != 1:
        log.error(f"Scene {s['id']} must have exactly one file, skipping...")
        return

    for file in scene['files']:
        scene_id = scene['id']
        path = file['path']
        file_id = file['id']
        fps = float(file['frame_rate'])
        dur = float(file['duration'])
        total_frames = int(dur * fps)
        log.debug(f'processing {scene_id=}...')
        endpoint = scene['stash_ids'][0]['endpoint']
        stash_id = scene['stash_ids'][0]['stash_id']

        cur = con.cursor()
        cur.execute("SELECT 1 FROM face WHERE endpoint = ? AND stash_id = ?",(endpoint, stash_id,))
        rows = cur.fetchall()
        if len(rows) > 0:
            log.info(f"face - skipping {scene_id=}, already processed")
            continue

        face_count = process_video(path, endpoint, stash_id)
#        cur.execute('INSERT INTO face (endpoint, stash_id, face_count, method) VALUES (?,?,?,?)',(endpoint, stash_id, face_count, METHOD,))
        log.debug(f"face - finished {scene_id=}")
        return con.commit()

def numpy_to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    return obj

def process_video(video_path, endpoint, stash_id, frequency=2, face_score_threshold=0.5):
    cur = con.cursor()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frequency)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    total_face_count = 0

    # Create a tqdm progress bar
    with tqdm(total=total_frames, desc="Processing video", disable=True) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                time_offset = round(frame_count / fps, 2)

                try:
                    # Analyze the frame
                    faces = model.get(frame)

                    total_face_count += len(faces)
                    total_face_area = 0.0

                    face_data = list()

                    for face in faces:
                        face_datum = process_face(face, frame, frame_count, face_score_threshold)
                        if face_datum == None:
                            continue
                        face_data.append(face_datum)
                        total_face_area += face_datum['bbox']['frame_fraction']


                    if len(face_data) > 0:
                        log.info(total_face_area)
                        d = json.dumps({
                            "count": len(faces),
                            "frame_fraction": total_face_area,
                            "faces": face_data,
                            "model": "buffalo_l",
                            "method": f"insightface-{insightface_version}",
                        })

                        cur.execute('INSERT INTO face (endpoint, stash_id, time_offset, faces, method) VALUES (?,?,?,?,?)',
                            (endpoint, stash_id, time_offset, json.dumps(numpy_to_python(d)), METHOD,)
                        )

                except Exception as e:
                    log.error(f"Error processing frame {frame_count}: {e}")

            frame_count += 1
            pbar.update(1)  # Update the progress bar

    cap.release()
    con.commit()
    log.info(f"Processed {frame_count} frames, detected {total_face_count} faces above threshold")
    return total_face_count

def process_face(face, frame, frame_offset, face_score_threshold):
    # Check face detection confidence
    if hasattr(face, 'det_score') and face.det_score < face_score_threshold:
        log.debug(f"Low confidence face detection at frame {frame_offset}, score: {face.det_score}")
        return None

    # Extract face attributes
    age = int(face.age)
    gender = 'M' if face.gender == 1 else 'F'
    embedding = face.embedding.tolist()

    # Save face image
    bbox = face.bbox.astype(int)
    real_coords, norm_coords, norm_face_coords, face_fraction = get_face_metrics(bbox, frame.shape)

    if face_fraction == 0:
        log.warning(f"Skipping empty face image at frame {frame_offset}")
        return None

    # Prepare JSON data
    face_data = {
        "age": age,
        "gender": gender,
        "bbox": {
            "real": numpy_to_python(real_coords),
            "normalized": numpy_to_python(norm_coords),
            "face_normalized": numpy_to_python(norm_face_coords),
            "frame_fraction": numpy_to_python(face_fraction),
        },
        "embedding": numpy_to_python(embedding),
        "det_score": numpy_to_python(face.det_score) if hasattr(face, 'det_score') else None
    }
    return face_data

def get_face_metrics(bbox, frame_shape):
    """
    Calculate various metrics for a detected face in a frame.
    
    Args:
    bbox (np.array): Bounding box coordinates [x1, y1, x2, y2]
    frame_shape (tuple): Shape of the frame (height, width, channels)
    
    Returns:
    tuple: (real_coords, norm_coords, norm_face_coords, face_fraction)
    """
    frame_height, frame_width = frame_shape[:2]
    
    # 1. Real coordinates (in pixels)
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(frame_width, bbox[2]), min(frame_height, bbox[3])
    real_coords = (x1, y1, x2, y2)
    
    # 2. Normalized coordinates of the face in the frame
    norm_x1, norm_y1 = x1 / frame_width, y1 / frame_height
    norm_x2, norm_y2 = x2 / frame_width, y2 / frame_height
    norm_coords = (norm_x1, norm_y1, norm_x2, norm_y2)
    
    # 3. Normalized coordinates of the actual face
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    norm_face_x1 = max(0, -bbox[0] / face_width)
    norm_face_y1 = max(0, -bbox[1] / face_height)
    norm_face_x2 = min(1, (frame_width - bbox[0]) / face_width)
    norm_face_y2 = min(1, (frame_height - bbox[1]) / face_height)
    norm_face_coords = (norm_face_x1, norm_face_y1, norm_face_x2, norm_face_y2)
    
    # 4. Fraction of the frame filled with the face
    face_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    face_fraction = face_area / frame_area
    
    return real_coords, norm_coords, norm_face_coords, face_fraction

def main():
    global stash
    json_input = json.loads(sys.stdin.read())
    FRAGMENT_SERVER = json_input["server_connection"]

    #log.debug(FRAGMENT_SERVER)

    stash = StashInterface(FRAGMENT_SERVER)
    PLUGIN_ARGS = False
    HOOKCONTEXT = False

    global con
    global face_dir_path
    face_db_path = sys.argv[1]
    face_dir_path = sys.argv[2]
    log.info(face_db_path)
    con = sqlite3.connect(face_db_path)

    try:
#        PLUGIN_ARGS = json_input['args'].get("mode")
#        PLUGIN_DIR = json_input["PluginDir"]
        PLUGIN_ARGS = json_input['args']["mode"]
    except:
        pass

    if PLUGIN_ARGS:
        log.debug("--Starting Plugin 'face'--")
        if "catchup" in PLUGIN_ARGS:
            log.info("Catching up with face extraction on older files")
            catchup() #loops thru all scenes, and tag
        exit_plugin("face plugin finished")

    try:
        HOOKCONTEXT = json_input['args']["hookContext"]
    except:
        exit_plugin("face hook: No hook context")

    log.debug("--Starting Hook 'face'--")


    sceneID = HOOKCONTEXT['id']
    scene = stash.find_scene(sceneID)

    results = checkface(scene)
    con.close()
    exit_plugin(results)

main()
