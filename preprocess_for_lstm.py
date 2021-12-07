"""This file converts data into sequence format and saves it"""

import os
import pickle

import pandas as pd

from LSTM_clean.utils import filter_and_split_data, printl, reindex_and_save_communities, sequence_generator

###################
### CONFIG
# TODO: User should init these values
LOAD_NAME = "twitch104k.csv"
LOAD_FOLDER = "/raid/home/myang349/recsys-filterbubbles/data/"
SAVE_FOLDER = "/raid/home/myang349/recsys-filterbubbles/data/twitch_sequence/"

# Use SAVE_PREFIX to mark different datasets
SAVE_PREFIX = "104k"
SAVE_TRAIN_NAME = SAVE_PREFIX + "train.data"
SAVE_VALID_NAME = SAVE_PREFIX + "valid.data"
SAVE_TEST_NAME = SAVE_PREFIX + "test.data"
SAVE_COMMUNITY_NAME = SAVE_PREFIX + "lstm_idx_communities.data"
USER_KEY = "user_id"
ITEM_KEY = "streamer_name"
TIME_KEY = "stop_time"
COMMUNITY_KEY = "community"
LOOK_BACK = 50


### 1. Reading File
print(f"Starting data retrieval for: {LOAD_FOLDER}")
df = pd.read_csv(os.path.join(LOAD_FOLDER, LOAD_NAME))
data = df[[USER_KEY, ITEM_KEY, TIME_KEY]].values


### 2. Preprocessing
print("\nPreprocessing Data into Sequence...")
unique_users = sorted(list(set(data[:, 0])))
unique_items = sorted(list(set(data[:, 1])))

data_with_splits = filter_and_split_data(data)
### NOTE: 1 is added to all the item IDs in this step
train, valid, test = sequence_generator(data_with_splits, look_back=LOOK_BACK)


### 3. Statistics
print(f"\nOriginal # of interactions: {len(df)}")
print(f"Post-Filtering # Of Interactions: {len(data_with_splits)}")
print(f"# of Training Points: {len(train)}")
print(f"# of Valid Points: {len(valid)}")
print(f"# of Test Points: {len(test)}")

### 4. Manipulate train, valid ,test
print(f"\nRe-indexing to fill in gaps")
lstm_idx_to_community, unique_items, item_to_lstm_idx, lstm_idx_to_df_item = reindex_and_save_communities(train, valid, test, df, ITEM_KEY, COMMUNITY_KEY)
print(f"Max item id in df: {max(lstm_idx_to_df_item.values())}")
print(f"Max itemid in data (before re-indexing): {max(item_to_lstm_idx.keys())}")
print(f"Max itemid in data (after re-indexing): {max(lstm_idx_to_df_item.keys())}")

### 5. Pickle the mapping info
print(f"\nPickling the community mapping...")
with open(os.path.join(SAVE_FOLDER, SAVE_COMMUNITY_NAME), "wb+") as f:
    pickle.dump(lstm_idx_to_community, f)

### 6. Pickle the datasets
print(f"\nPickling the re-indexed datasets...")
with open(os.path.join(SAVE_FOLDER, SAVE_TRAIN_NAME), "wb+") as f:
    pickle.dump(train, f)
with open(os.path.join(SAVE_FOLDER, SAVE_VALID_NAME), "wb+") as f:
    pickle.dump(valid, f)
with open(os.path.join(SAVE_FOLDER, SAVE_TEST_NAME), "wb+") as f:
    pickle.dump(test, f)
print(f"Done!")
