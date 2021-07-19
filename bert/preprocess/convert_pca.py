#!/home/mtran/anaconda3/bin/python
import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
import random
import pickle
import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Manager
import subprocess

# input_path = "/shares/perception-temp/voxceleb2/facemesh/train/"
# output_path = "/shares/perception-working/minh/transformer_data/facemesh_pca6/train/"
input_path, output_path = sys.argv[1], sys.argv[2]
model_path = "../data/pca_6.pkl"
pca_model = pickle.load(open(model_path, 'rb'))


def rescale(landmarks):
    x = landmarks[:, 0:468]
    y = landmarks[:, 468:468*2]
    z = landmarks[:, 468*2:468*3]

    processed_x = []
    processed_y = []
    processed_z = []

    for i in range(x.shape[0]):
        z[i] -= z[i].mean()
        current_x, current_y, current_z = x[i], y[i], z[i]
        scale = 1.0 / (max(current_y)-min(current_y))
        current_x = current_x * scale * (-1)
        current_y = current_y * scale * (-1)
        # plt.scatter(current_x, current_y, s = 4)
        # plt.show()
        processed_x.append(current_x)
        processed_y.append(current_y)
        processed_z.append(current_z)
    return np.concatenate((np.array(processed_x), np.array(processed_y), np.array(processed_z)), axis=1)


def convert(pca_model, file_paths):
    for file in file_paths:
        data = rescale(pd.read_csv(file, header=None).values)
        transformed_data = pca_model.transform(data)
        file_out = os.path.join(output_path, file.split('/')[-1])
        pd.DataFrame(transformed_data).to_csv(file_out, header=None, index=False)


def convert_in_parallel(concurreny_count, files, fn):
    Processes = []
    files_ = [files[(i * (len(files)//concurreny_count)):((i+1) * (len(files)//concurreny_count))]
              for i in range(concurreny_count)]
    leftovers = files[(concurreny_count * (len(files)//concurreny_count)):  len(files)]
    for i in range(len(leftovers)):
        files_[i] += [leftovers[i]]

    for files_list_ in files_:
        p = Process(target=fn, args=(pca_model, files_list_))
        Processes.append(p)
        p.start()
    # block until all the threads finish (i.e. block until all function_x calls finish)
    for t in Processes:
        t.join()


file_list = os.listdir(input_path)
convert_in_parallel(200, file_list, convert)
