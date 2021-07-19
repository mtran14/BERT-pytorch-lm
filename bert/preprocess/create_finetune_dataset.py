#!/home/mtran/anaconda3/bin/python
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.cluster import KMeans
import random
from sklearn.preprocessing import LabelEncoder


def mosi_filename_convert(input_string):
    # convert input string such that last 4 index are digits
    cnt = 0
    output_str = ''
    for i in reversed(range(len(input_string))):
        if(input_string[i].isdigit()):
            cnt += 1
            output_str += input_string[i]
        else:
            if(cnt < 4):
                output_str += '0'*(4-cnt)
                cnt = 4
            output_str += input_string[i]
    return output_str[::-1]


centroids_video = pd.read_csv('~/transformer/data/centroids_facemesh_5000.csv', header=None).values
kmeans_video = KMeans(n_clusters=centroids_video.shape[0])
kmeans_video.cluster_centers_ = centroids_video

label_dict = {}  # map filename -> corresponding label
input_path_train_video, label_file, data_type = sys.argv[1], sys.argv[2], sys.argv[3]
meta_data_df = pd.read_csv(label_file)
meta_data = meta_data_df.values
X, y, group = [], [], []

if(data_type == 'meld'):
    # do something
    pass
elif(data_type == 'cremad'):
    label_list = []
    for row in meta_data:
        if(len(row[10]) == 1):
            label_list.append(row[10])
    le = LabelEncoder().fit(label_list)
    for row in meta_data:
        if(len(row[10]) == 1):
            label = le.transform([row[10]])[0]
        label_dict[row[7]] = [label, row[7][:4]]
        group.append(row[7][:4])

elif(data_type == 'mosi'):
    for row in meta_data:
        label_dict[row[0]] = [row[-1], row[0][:-5]]
        group.append(row[0][:-5])

group = list(set(group))
train_group, val_group, test_group = [], [], []
for pid in group:
    r = random.uniform(0, 1)
    if(r < 0.6):
        train_group.append(pid)
    elif(r < 0.8):
        val_group.append(pid)
    else:
        test_group.append(pid)

output_train, output_test, output_val = [], [], []
for file in os.listdir(input_path_train_video):
    current_data = pd.read_csv(os.path.join(input_path_train_video, file), header=None).values
    current_cluster = kmeans_video.predict(current_data)
    current_str = ''
    for i in range(len(current_cluster)):
        current_str += str(current_cluster[i]) + ' '
    label = label_dict[mosi_filename_convert(file.split('.')[0])][0]
    pid = mosi_filename_convert(file.split('.')[0])[:-5]
    if(pid in train_group):
        output_train.append([current_str.strip(), label])
    elif(pid in val_group):
        output_val.append([current_str.strip(), label])
    else:
        output_test.append([current_str.strip(), label])

pd.DataFrame(output_train).to_csv(data_type + '_train.tsv',
                                  sep='\t', index=False, header=['sentence', 'label'])
pd.DataFrame(output_val).to_csv(data_type + '_dev.tsv',
                                sep='\t', index=False, header=['sentence', 'label'])
pd.DataFrame(output_test).to_csv(data_type + '_test.tsv',
                                 sep='\t', index=False, header=['sentence', 'label'])
