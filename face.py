import os
import sys
import json
import sqlite3

METHOD = 'face-1.0.0'

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

        face_count = process_video(path, f"{face_dir_path}/{stash_id}")
        cur.execute('INSERT INTO face (endpoint, stash_id, face_count, method) VALUES (?,?,?,?)',(endpoint, stash_id, face_count, METHOD,))
        log.debug(f"face - finished {scene_id=}")
        return con.commit()

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def process_video(video_path, output_dir, frequency=2, face_score_threshold=0.5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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
                try:
                    # Analyze the frame
                    faces = model.get(frame)

                    # Reset face index for each processed frame
                    face_index = 0

                    for face in faces:
                        # Check face detection confidence
                        if hasattr(face, 'det_score') and face.det_score < face_score_threshold:
                            log.debug(f"Low confidence face detection at frame {frame_count}, score: {face.det_score}")
                            continue

                        # Extract face attributes
                        age = int(face.age)
                        gender = 'M' if face.gender == 1 else 'F'
                        embedding = face.embedding.tolist()

                        # Save face image
                        bbox = face.bbox.astype(int)
                        face_img = frame[max(0, bbox[1]):min(frame.shape[0], bbox[3]),
                                         max(0, bbox[0]):min(frame.shape[1], bbox[2])]

                        if face_img.size == 0:
                            log.warning(f"Skipping empty face image at frame {frame_count}")
                            continue

                        timestamp = timedelta(seconds=frame_count/fps)
                        base_filename = f"T{timestamp.total_seconds():05.2f}_{gender}{age}_{face_index}"
                        jpg_filename = f"{base_filename}.jpg"
                        json_filename = f"{base_filename}.json"
                        jpg_filepath = os.path.join(output_dir, jpg_filename)
                        json_filepath = os.path.join(output_dir, json_filename)

                        try:
                            cv2.imwrite(jpg_filepath, face_img)
                        except cv2.error as e:
                            log.error(f"Error saving face image: {e}")
                            continue

                        # Prepare JSON data
                        face_data = {
                            "age": age,
                            "gender": gender,
                            "embedding": embedding,
                            "model": "buffalo_l",
                            "method": f"insightface-{insightface_version}",
                            "det_score": numpy_to_python(face.det_score) if hasattr(face, 'det_score') else None
                        }

                        # Save JSON file
                        with open(json_filepath, 'w') as json_file:
                            json.dump(face_data, json_file)

                        # Increment face index for this frame
                        face_index += 1
                        # Increment total face count
                        total_face_count += 1

                except Exception as e:
                    log.error(f"Error processing frame {frame_count}: {e}")

            frame_count += 1
            pbar.update(1)  # Update the progress bar

    cap.release()
    log.info(f"Processed {frame_count} frames, detected {total_face_count} faces above threshold")
    return total_face_count


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
