# Face detection for Stashapp

This plugin uses the Python `InsightFace` model `buffalo_l` to extract face
embeddings from all scenes at 2Hz

## Purpose

Performer tagging.

## How to configure the plugin

0. Install requirements: `pip install -r requirements.txt`. Briefly, it's
   opencv, stashapp-tools, perception, and their respective dependencies.
   Tested with Python 3.10

1. Create a database for storing perceptual hashes:
  ```
  echo "
    CREATE TABLE face (
      endpoint TEXT NOT NULL,
      stash_id TEXT NOT NULL,
      face_count INT NOT NULL,
      method TEXT NOT NULL, 
      UNIQUE (stash_id, method)
    );
  " | sqlite3 /path/to/face.sqlite

2. Update `face.yml` to use the path to the sqlite datbase you created. In the
   config, it's by default: `  - "{pluginDir}/../face.sqlite"`
  Change accordingly.

3. Similarly, if you want to change the output directory for face images and
   json embeddings, do that too. By default they're under `generated` with vtt
   files, etc.

## How to use the plugin

In stashapp settings > tasks, under plugin tasks, find a new task labeled `Face
extract scenes`. This will trigger a database-wide operation. It may take many
days to complete. Don't worry about interrupting it, it only commits to its
database after processing a file, so interruption won't be a problem - you can
resume quickly anytime and without losing progress.
