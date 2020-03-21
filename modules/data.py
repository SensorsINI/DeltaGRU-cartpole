import numpy
import torch
from torch.utils import data
import pandas as pd
import numpy as np
import math
from scipy.signal import medfilt as medfilt

# TODO check value, might not be 4096 full scale. It does rotate all the way, is 0 at left, 2000 to right, about 3210 vertical up, and -947 hanging vertically
RAD_PER_ANGLE_ADC = 2 * math.pi / 4096  # a complete rotation of the potentiometer is mapped to this many ADC counts
POSITION_LIMIT=4000 # position encoder has limit +/- POSITION_MAX

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


def load_data(filepath, cw_plen, cw_flen, pw_len, pw_off, seq_len, stride=1, medFilterWindow=3):
    '''
    Loads dataset from CSV file
    Args:
        filepath: path to CSV file
        cw_plen: context window
        cw_flen: TODO
        pw_len: prediction window
        pw_off: prediction offset
        seq_len: number of samples in time input to RNN
        stride: step over data in samples between samples
        medFilterWindow: median filter window, 0 to disable

    Returns:
        Normalized torch.Tensor(data).float() tensors
        input_data:  indexed by [sample, sequence, # sensor inputs * cw_len]
        label_data:  indexed by [sample, sequence, # output sensor values * pw_len]
        mean_train_data, std_train_data, mean_target_data, std_target_data: the means and stds of training and target data

        The last dimension of input_data and label_data is organized by all sensors for sample0, all sensors for sample1, .....all sensors for sampleN
        Eech sensor dimension is indpendently normalized by it mean and std so that it is zero centered with std=1
    '''

    # Load dataframe
    df = pd.read_csv(filepath)

    # time,deltaTimeMs,angle,position,angleTarget,angleErr,positionTarget,positionErr,angleCmd,positionCmd,motorCmd,actualMotorCmd
    # 172.2985134124756,4.787921905517578,1699,-418,3129,-1428.0,0,-418.0,573146.4030813494,-8360.0,7055,0
    # Collect data
    train_data = []
    all_data = []

    time=df.time.to_numpy()
    deltaTimeMs=df.deltaTimeMs.to_numpy()

    # Get Raw Data
    angle = df.angleErr.to_numpy() # simplify the pole angle to just angle error, so it is zero centered around presumed vertical position
    # desired vertical angle might be incorrectly set, resulting in large position offset that balances position and angle control
    angle = angle*RAD_PER_ANGLE_ADC # angle has range -pi to +pi, centered very closely around 0 during balancing
    if medFilterWindow>0:
        angle=medfilt(angle,medFilterWindow)
    position = df.position.to_numpy()/POSITION_LIMIT # leave cart position as absolute position since the target position might change with time and the values
                                        # for control might depend on position of cart
    # position has range < +/-1
    if medFilterWindow>0:
        position=medfilt(position,medFilterWindow)

    # project angle onto x and y since angle is a rotational variable with 2pi cut that we cannot fit properly and does not represent gravity and lateral acceleration well.
    sinAngle=np.sin(angle)
    cosAngle=np.cos(angle)
    # compute temporal derivatives from state data
    averageDeltaTMs=deltaTimeMs.mean() # TODO this is approximate derivative since sample rate varied a bit around 5ms
    # TODO consider using a better controlled derivative that enforces e.g. total variation constraint
    # derivative is compute

    # dAngle=np.gradient(angle,averageDeltaTMs)
    actualMotorCmd = df.actualMotorCmd.to_numpy() # zero-centered motor speed command
    
    # Min-Max Normalize TODO don't do this since main.py already normalized training and labels for training and test using mean/std
    # def normalize_min_max(x, x_min, x_max):
    #     return (x - x_min)/(x_max - x_min)
    # angle_min = np.amin(angle)
    # angle_max = np.amax(angle)
    # angle = normalize_min_max(angle, angle_min, angle_max)
    # position_min = np.amin(position)
    # position_max = np.amax(position)
    # position = normalize_min_max(position, position_min, position_max)
    # actualMotorCmd_min = np.amin(actualMotorCmd)
    # actualMotorCmd_max = np.amax(actualMotorCmd)
    # actualMotorCmd = normalize_min_max(actualMotorCmd, actualMotorCmd_min, actualMotorCmd_max)
    
    # Derive Other Data
    # dAngle = np.gradient(angle,averageDeltaTMs)
    # ddAngle = np.gradient(dAngle, edge_order=1)  # same for accelerations
    dCosAngle = np.gradient(cosAngle,averageDeltaTMs)
    dSinAngle = np.gradient(sinAngle,averageDeltaTMs)
    ddCosAngle = np.gradient(dCosAngle, averageDeltaTMs)
    ddSinAngle = np.gradient(dSinAngle, averageDeltaTMs)
    dPosition = np.gradient(position, averageDeltaTMs)
    ddPosition = np.gradient(dPosition, edge_order=1)

    # Data  We never use this and it consumes a lot of memory
    # all_data.append(time)
    # all_data.append(sinAngle)
    # all_data.append(cosAngle)
    # all_data.append(dSinAngle)
    # all_data.append(dCosAngle)
    # all_data.append(ddSinAngle)
    # all_data.append(ddCosAngle)
    # all_data.append(position)
    # all_data.append(dPosition)
    # all_data.append(ddPosition)
    # all_data.append(actualMotorCmd)

    # Train Data
    # train_data.append(angle)
    # train_data.append(dAngle)
    # train_data.append(ddAngle)
    train_data.append(sinAngle)
    train_data.append(cosAngle)
    train_data.append(dSinAngle)
    train_data.append(dCosAngle)
    # train_data.append(ddSinAngle)
    # train_data.append(ddCosAngle)
    train_data.append(position)
    train_data.append(dPosition)
    # train_data.append(ddPosition)
    train_data.append(actualMotorCmd)

    # all_data = np.vstack(all_data).transpose()
    train_data = np.vstack(train_data).transpose()   # train_data indexed by [sample, input sensor/control]

    # Collect desired prediction
    # All terms are weighted equally by the loss function (either L1 or L2),
    # so we need to make sure that we don't have values here that are too different in range
    # Since the derivatives are computed on normalized data, but divided by the delta time in ms,
    # we need to normalize the derivatives too. (We include the delta time to make more physical units of time
    # so that the values don't change with different sample rate.)

    target = []
    target.append(sinAngle)
    target.append(cosAngle)
    target.append(dSinAngle)
    target.append(dCosAngle)
    # target.append(angle)
    # target.append(dAngle)
    target.append(position)
    target.append(dPosition)
    target = np.vstack(target).transpose()   # target indexed by [sample, sensor]

    # Get sample
    # test_sample = all_data # indexed by [sampleNumber, dataType]
    # print("Test Sample Size: ", ampro_test_sample.shape)
    train_sample = train_data[1:-1, :]  # TODO why throw away first and last time point, maybe CSV file incomplete
    # print("Sample Size: ", train_data.shape)
    # print("Train Max: ", np.amax(train_sample))
    # print("Train Min: ", np.amin(train_sample))
    # print("Test Max: ", np.amax(train_sample))
    # print("Test Min: ", np.amin(train_sample))

    # Print Collected Data Information
    print('###################################################################################\n\r'
          '# Collected Data Information\n\r'
          '###################################################################################')
    print("train_sample size: ", train_sample.shape)

    # Split train data
    numSamples = train_sample.shape[0]

    data_new = []
    target_new = []
    # Iterate from the current timestep to the last timestep that gives the last prediction window
    for i in range(cw_plen, numSamples - max(cw_flen, pw_len+pw_off) - seq_len):
        if i % stride is not 0:
            continue
        data_seq = []
        label_seq = []
        for t in range(0, seq_len):
            data_seq.append(train_sample[i+t-cw_plen:i+t+cw_flen+1, :].ravel())
            # ravel() makes 1d array for sensor values and prediction window length
            # order is row major, with last index changing fastest
            # since last index is sensor, the order is [sensor1t0, sensor2t0, ... sensorNt0, sensor0t1, sensor2t1 ....]
            label_seq.append(target[i+t+pw_off:i+t+pw_off+pw_len, :].ravel())
        data_seq = np.stack(data_seq, axis=0)
        label_seq = np.stack(label_seq, axis=0)
        data_new.append(data_seq)
        target_new.append(label_seq)
    data_new = np.stack(data_new, axis=0) # indexed by [sample, sequence, # sensor inputs * cw_len]
    target_new = np.stack(target_new, axis=0) # [sample, sequence, # output sensor values * pw_len]

    print("data_new size: ", data_new.shape)
    print("target_new size: ", target_new.shape)
    # print("Postion Min: ", position_min)
    # print("Postion Min: ", position_max)
    # print("Angle Min:   ", angle_min)
    # print("Angle Max:   ", angle_max)

    # To compute normalization,
    # reshape the data_new tensor from [sample, sequence, # sensor inputs * cw_len] to [sample*sequence, # sensor inputs*cw_len]
    # torch.mean(data_new, 0)
    # calculates the mean of elements along the first dimension
    # The result is a [#sensor_inputs*cw_len] vector containing the mean of every feature in the data_new

    

    # Convert Numpy Arrays to PyTorch Tensors
    data_new = torch.from_numpy(data_new).float()
    target_new = torch.from_numpy(target_new).float()

    n = target_new.size(0) * target_new.size(1)
    mean_target_data = torch.mean(target_new.reshape(n, -1), 0)
    std_target_data = torch.std(target_new.reshape(n, -1), 0)
    label_data_norm = (target_new - mean_target_data) / std_target_data
    
    n=data_new.size(0) * data_new.size(1)
    mean_train_data = torch.mean(data_new.reshape(n, -1), 0)
    std_train_data = torch.std(data_new.reshape(n, -1), 0)
    train_data_norm = (data_new - mean_train_data) / std_train_data

    return train_data_norm, label_data_norm, mean_train_data, std_train_data, mean_target_data, std_target_data
    # return test_sample, data_new, target_new
