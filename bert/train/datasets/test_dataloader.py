import torch
from pretraining import PairedDatasetC
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('/home/mtran/BERT-pytorch-lm/bert/train/')
from utils.collate import pretraining_collate_functionC

path = "/home/mtran/toydata/"
mdataset = PairedDatasetC(path)
dataloader = DataLoader(mdataset, batch_size=4, shuffle=True, collate_fn=pretraining_collate_functionC)

for inputs, targets, batch_count in tqdm(dataloader):
    sequence, segment = inputs
    target, isnext = targets
    assert (sequence.shape == target.shape)
print('ok.')
