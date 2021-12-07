import numpy as np
import pickle
import os

cwd = os.getcwd()
test_path = cwd + "/data/test.data"
train_path = cwd + "/data/train.data"
valid_path = cwd + "/data/valid.data"

train_data = np.load(train_path, allow_pickle=True)
valid_data = np.load(valid_path, allow_pickle=True)
# test_data = np.load(path, allow_pickle=True)