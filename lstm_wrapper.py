import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from copy import deepcopy

from LSTM_clean.model import LSTM

# Data Location
SAVE_FOLDER = "/raid/home/myang349/recsys-filterbubbles/data/twitch_sequence/"
SAVE_TRAIN_NAME = "train.data"
SAVE_VALID_NAME = "valid.data"
SAVE_TEST_NAME = "test.data"

# Configuration for MODEL
EPOCHS = 200
# Should be # of unique items in data + 1 for the 0 item
OUTPUT_SIZE = 5400
LEARNING_RATE = 5e-3
MOMENTUM = 0.9

def train_model():
    # Setting Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is", device)

    # The format is:
    # N x 2 x (sequence, 
    train_data = np.load(os.path.join(SAVE_FOLDER, SAVE_TRAIN_NAME), allow_pickle=True)
    valid_data = np.load(os.path.join(SAVE_FOLDER, SAVE_VALID_NAME), allow_pickle=True)
    test_data = np.load(os.path.join(SAVE_FOLDER, SAVE_TEST_NAME), allow_pickle=True)

    print(f"Train: {len(valid_data)}, Valid: {len(valid_data)}")

    # Output size should be # of unique items in data + 1 for the 0 item
    model = LSTM(input_size=128, output_size=OUTPUT_SIZE, hidden_dim=64, n_layers=1, device=device).to(device)
    model.LSTM.flatten_parameters()
    print("Model is ", model)

    print("\nTraining and testing")
    model.traintest(train=train_data,test=valid_data, epochs=EPOCHS, learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    print("\nFinished!")

    return model

def get_topk_predictions(model, data, k):
    data = deepcopy(data)

    # Generate embeddings and move to cuda
    embedded_data = []
    for pt in data:
        embedded_data.append(model.item_emb(torch.LongTensor(pt).to(model.device)))
    embedded_data = torch.stack(embedded_data, dim=0).detach()

    output, hidden = model.forward(embedded_data)
    preds = torch.topk(output, k).indices.tolist()
    return preds