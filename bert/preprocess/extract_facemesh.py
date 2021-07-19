#!/home/mtran/anaconda3/bin/python
import cv2
import math
import numpy as np
import mediapipe as mp
import os
import sys
from multiprocessing import Process, Manager
import pandas as pd
import random
import subprocess
import time
from moviepy.editor import VideoFileClip

# os.chdir("/home/mtran/OpenFace/build/")

# output_path = "/shares/perception-temp/voxceleb2/facemesh/train/"
# output_path = "/home/mtran/Downloads/emotion_facemesh/meld/val/"
# path = "/home/mtran/Downloads/MELD/dev_splits_complete/"

output_path, path = sys.argv[2], sys.argv[1]


def extractLandMarks(fm_results):
    """TODO: add preprocessing/normalization step here"""
    x = []
    y = []
    z = []
    for i in range(468):
        x.append(fm_results.multi_face_landmarks[0].landmark[i].x)
        y.append(fm_results.multi_face_landmarks[0].landmark[i].y)
        z.append(fm_results.multi_face_landmarks[0].landmark[i].z)
    return x + y + z


def facemesh_extract(files, buff):
    mp_face_mesh = mp.solutions.face_mesh
    random.shuffle(files)
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        for file in files:
            try:
                file_path_split = file.split("/")
                # id1, id2, fname = file_path_split[-3], file_path_split[-2], file_path_split[-1]
                output_file_name = file_path_split[-1].split('.')[0] + '.csv'
                output_file_path = os.path.join(output_path, output_file_name)
                if(os.path.isfile(output_file_path)):
                    continue
                vidcap = VideoFileClip(file)
                frames = list(vidcap.iter_frames(fps=5))

                output = []
                for frame in frames:
                    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if not results.multi_face_landmarks:
                        continue
                    output.append(list(extractLandMarks(results)))

                if(len(output) > 0):
                    pd.DataFrame(output).to_csv(output_file_path, header=None, index=False)
                vidcap.close()
            except:
                print(file)
                continue


def facemesh_extract_in_parallel(concurreny_count, files, fn):
    Processes = []
    files_ = [files[(i * (len(files)//concurreny_count)):((i+1) * (len(files)//concurreny_count))]
              for i in range(concurreny_count)]
    leftovers = files[(concurreny_count * (len(files)//concurreny_count)):  len(files)]
    for i in range(len(leftovers)):
        files_[i] += [leftovers[i]]

    for files_list_ in files_:
        p = Process(target=fn, args=(files_list_, files_list_))
        Processes.append(p)
        p.start()
    # block until all the threads finish (i.e. block until all function_x calls finish)
    for t in Processes:
        t.join()


concurreny_count = 5
# path = "/data/perception-working/minh/ted/VIDEO/"
# path = "/shares/perception-temp/voxceleb2/train/dev/mp4/"


files = []
for file in os.listdir(path):
    files.append(os.path.join(path, file))
print(files[0:3])
facemesh_extract_in_parallel(concurreny_count, files, facemesh_extract)
