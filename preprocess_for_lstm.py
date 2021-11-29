"""This file converts data into sequence format and saves it"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from tracin.tracin import (
    save_tracin_checkpoint,
    load_tracin_checkpoint,
    calculate_tracin_influence,
)
import pandas as pd
from LSTM_clean.utils import train_test_split, sequence_generator, printl
from LSTM_clean.model import LSTM
from collections import Counter
import numpy as np
import pickle

###################
### CONFIG
# TODO: User should init these values
LOAD_FOLDER = "/raid/home/myang349/recsys-filterbubbles/data/"
LOAD_NAME = "twitch100k.csv"
SAVE_FOLDER = "/raid/home/myang349/recsys-filterbubbles/data/twitch_sequence/"


SAVE_TRAIN_NAME = "train.data"
SAVE_VALID_NAME = "valid.data"
SAVE_TEST_NAME = "test.data"
USER_KEY = "user_id"
ITEM_KEY = "streamer_name"
TIME_KEY = "stop_time"
LOOK_BACK = 50


### 1. Reading File
print(f"Starting data retrieval for: {LOAD_FOLDER}")
df = pd.read_csv(os.path.join(LOAD_FOLDER, LOAD_NAME))
data = df[[USER_KEY, ITEM_KEY, TIME_KEY]].values


### 2. Preprocessing
print("\nPreprocessing Data into Sequence...")
unique_users = sorted(list(set(data[:, 0])))
unique_items = sorted(list(set(data[:, 1])))

original_data = train_test_split(data)
train, valid, test = sequence_generator(data, LOOK_BACK)

### 3. Statistics
print(f"\nOriginal # of interactions: {len(df)}")
print(f"# of Training Points: {len(train)}")
print(f"# of Valid Points: {len(valid)}")
print(f"# of Test Points: {len(test)}")

### 4. Pickle
print(f"\nPickling...")
with open(os.path.join(SAVE_FOLDER, SAVE_TRAIN_NAME), "wb") as f:
    pickle.dump(train, f)
with open(os.path.join(SAVE_FOLDER, SAVE_VALID_NAME), "wb") as f:
    pickle.dump(valid, f)
with open(os.path.join(SAVE_FOLDER, SAVE_TEST_NAME), "wb") as f:
    pickle.dump(test, f)
print(f"Done!")
