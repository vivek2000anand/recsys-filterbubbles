import numpy as np
import pickle
import os

cwd = os.getcwd()
path = cwd + "/data/test.data"
test_data = np.load(path, allow_pickle=True)
print(test_data)
# valid_data = np.load(path, allow_pickle=True)
# test_data = np.load(path, allow_pickle=True)