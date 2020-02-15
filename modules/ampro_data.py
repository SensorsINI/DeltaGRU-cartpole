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


def load_ampro_data(filepath, look_back_len, pred_len):
    # Load dataframe
    df = pd.read_csv(filepath)

    # Collect data
    train_data = []
    spi_data = []
    # ampro_data.append(df.force_toe.to_numpy())
    # ampro_data.append(df.tau.to_numpy())
    # ampro_data.append(df.position_actual_ankle.to_numpy())
    # ampro_data.append(df.position_actual_knee.to_numpy())
    # ampro_data.append(df.velocity_actual_ankle.to_numpy())
    # ampro_data.append(df.velocity_actual_knee.to_numpy())
    # ampro_data.append(np.gradient(df.velocity_actual_ankle.to_numpy(), edge_order=1))
    # ampro_data.append(np.gradient(df.velocity_actual_knee.to_numpy(), edge_order=1))

    ishumanswing = df.ishumanswing.to_numpy()
    position_actual_ankle = df.position_actual_ankle.to_numpy()
    position_actual_knee = df.position_actual_knee.to_numpy()
    velocity_actual_ankle = df.velocity_actual_ankle.to_numpy()
    velocity_actual_knee = df.velocity_actual_knee.to_numpy()
    # torque_actual_ankle = df.torque_actual_ankle.to_numpy()
    # torque_actual_knee = df.torque_actual_knee.to_numpy()
    # position_desired_ankle = df.position_desired_ankle.to_numpy()
    position_desired_knee = df.position_desired_knee.to_numpy()
    # velocity_desired_ankle = df.velocity_desired_ankle.to_numpy()
    velocity_desired_knee = df.velocity_desired_knee.to_numpy()
    position_error_knee = position_actual_knee - position_desired_knee
    velocity_error_knee = velocity_actual_knee - velocity_desired_knee


    # acceleration_actual_ankle = np.gradient(df.velocity_actual_ankle.to_numpy(), edge_order=1)
    # acceleration_actual_knee = np.gradient(df.velocity_actual_knee.to_numpy(), edge_order=1)
    # acceleration_desired_knee = np.gradient(df.velocity_desired_knee.to_numpy(), edge_order=1)
    # acceleration_error_knee = acceleration_actual_knee - acceleration_desired_knee

    # SPI Data
    spi_data.append(ishumanswing)
    spi_data.append(position_actual_knee)
    spi_data.append(position_desired_knee)
    spi_data.append(velocity_actual_knee)
    spi_data.append(velocity_desired_knee)
    spi_data.append(position_actual_ankle)
    spi_data.append(velocity_actual_ankle)

    # Train Data
    train_data.append(ishumanswing)
    train_data.append(position_error_knee)
    train_data.append(velocity_error_knee)
    train_data.append(position_actual_ankle)
    train_data.append(velocity_actual_ankle)


    # ampro_data.append(position_actual_ankle)  # SPI
    # ampro_data.append(position_actual_knee)   # SPI
    # ampro_data.append(position_desired_knee)  # SPI
    # ampro_data.append(velocity_actual_ankle)  # SPI
    # ampro_data.append(velocity_actual_knee)   # SPI
    # ampro_data.append(velocity_desired_knee)  # SPI

    # ampro_data.append(acceleration_actual_ankle)
    # ampro_data.append(acceleration_actual_knee)
    # ampro_data.append(acceleration_desired_knee)
    # ampro_data.append(acceleration_error_knee)
    # ampro_data.append(torque_actual_ankle)
    # ampro_data.append(torque_actual_knee)


    # print(np.isnan(df.position_desired_ankle.to_numpy()).any())
    # print(np.isnan(df.position_desired_knee.to_numpy()).any())
    # print(np.isnan(df.velocity_desired_ankle.to_numpy()).any())
    # print(np.isnan(df.velocity_desired_knee.to_numpy()).any())
    # print(np.isnan(np.gradient(df.velocity_desired_ankle.to_numpy(), edge_order=1)).any())
    # print(np.isnan(np.gradient(df.velocity_desired_knee.to_numpy(), edge_order=1)).any())

    spi_data = np.vstack(spi_data).transpose()
    train_data = np.vstack(train_data).transpose()

    # Collect label
    labels = []
    labels.append(df.torque_desired_ankle.to_numpy())
    labels.append(df.torque_desired_knee.to_numpy())
    # labels.append(df.control_signal_ankle.to_numpy())
    # labels.append(df.control_signal_knee.to_numpy())
    labels = np.vstack(labels).transpose()

    # Get AMPRO sample
    ampro_test_sample = spi_data
    # print("Ampro Test Sample Size: ", ampro_test_sample.shape)
    ampro_train_sample = train_data[1:-1, :]
    # print("Ampro Sample Size: ", train_data.shape)
    # print("Ampro Train Max: ", np.amax(ampro_train_sample))
    # print("Ampro Train Min: ", np.amin(ampro_train_sample))
    # print("Ampro Test Max: ", np.amax(ampro_train_sample))
    # print("Ampro Test Min: ", np.amin(ampro_train_sample))

    # Split train data
    train_data_len = ampro_train_sample.shape[0]

    data_new = []
    label_new = []
    for i in range(0, train_data_len - pred_len - look_back_len):
        data_new.append(ampro_train_sample[i:i + look_back_len, :])
        label_new.append(labels[i + pred_len:i + look_back_len + pred_len])
    data_new = np.stack(data_new, axis=0)
    label_new = np.stack(label_new, axis=0)

    return ampro_test_sample, data_new, label_new
