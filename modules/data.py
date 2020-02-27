import numpy
import torch
from torch.utils import data
import pandas as pd
import numpy as np
class Dataset(data.Dataset):
    def __init__(self, data, labels, retrain):
        'Initialization'
        self.data = data
        self.labels = labels
        self.retrain = retrain

    def __len__(self):
        'Total number of samples'
        return self.data.size(0)  # The first dimention of the data tensor

    def __getitem__(self, idx):
        'Get one sample from the dataset using an index'
        if self.retrain == 3:
            X = self.data[idx, :]
        else:
            X = self.data[idx, :, :]
        y = self.labels[idx, :]

        return X, y


def load_data(filepath, look_back_len, pred_len):
    # Load dataframe
    df = pd.read_csv(filepath)

    # time,deltaTimeMs,angle,position,angleTarget,angleErr,positionTarget,positionErr,angleCmd,positionCmd,motorCmd,actualMotorCmd
    # 172.2985134124756,4.787921905517578,1699,-418,3129,-1428.0,0,-418.0,573146.4030813494,-8360.0,7055,0
    # Collect data
    train_data = []
    all_data = []

    time=df.time.to_numpy()
    deltaTimeMs=df.deltaTimeMs.to_numpy()

    angle = df.angleErr.to_numpy() # simplify the angle to just angle error, so it is zero centered
    position = df.positionErr.to_numpy() # same for position
    # compute temporal derivatives from state data
    averageDeltaT=deltaTimeMs.mean() # TODO this is approximate derivative since sample rate varied a bit around 5ms
    dAngle=np.gradient(angle,averageDeltaT)
    dPosition=np.gradient(position,averageDeltaT)
    actualMotorCmd = df.actualMotorCmd.to_numpy() # zero-centered motor speed command

     # Data
    all_data.append(time)
    all_data.append(angle)
    all_data.append(dAngle)
    all_data.append(position)
    all_data.append(dPosition)
    all_data.append(actualMotorCmd)

    # Train Data
    train_data.append(angle)
    train_data.append(dAngle)
    train_data.append(position)
    train_data.append(dPosition)
    train_data.append(actualMotorCmd)

    all_data = np.vstack(all_data).transpose()
    train_data = np.vstack(train_data).transpose()

    # Collect label
    labels = []
    labels.append(angle)
    labels.append(dAngle)
    labels.append(position)
    labels.append(dPosition)
    labels = np.vstack(labels).transpose()

    # Get sample
    test_sample = all_data
    # print("Test Sample Size: ", ampro_test_sample.shape)
    train_sample = train_data[1:-1, :]
    # print("Sample Size: ", train_data.shape)
    # print("Train Max: ", np.amax(train_sample))
    # print("Train Min: ", np.amin(train_sample))
    # print("Test Max: ", np.amax(train_sample))
    # print("Test Min: ", np.amin(train_sample))

    # Split train data
    train_data_len = train_sample.shape[0]

    data_new = []
    label_new = []
    for i in range(0, train_data_len - pred_len - look_back_len):
        data_new.append(train_sample[i:i + look_back_len, :])
        label_new.append(labels[i + pred_len:i + look_back_len + pred_len])
    data_new = np.stack(data_new, axis=0)
    label_new = np.stack(label_new, axis=0)

    return test_sample, data_new, label_new
