import numpy
import torch
from torch.utils import data
import pandas as pd
import numpy as np
import math
from scipy.signal import butter, lfilter, lfilter_zi, medfilt
import logging

# TODO check value, might not be 4096 full scale. It does rotate all the way, is 0 at left, 2000 to right, about 3210 vertical up, and -947 hanging vertically
RAD_PER_ANGLE_ADC = 2 * math.pi / 4096  # a complete rotation of the potentiometer is mapped to this many ADC counts
POSITION_LIMIT = 4000  # position encoder has limit +/- POSITION_MAX
MEDFILT_WINDOW=5
GRADIENT_ORDER = 2  # type of gradient, 1 or 2 order
CUTOFF_HZ = 10  # Hz
FS = 200  # Hz sample rate


def filterAndGradients(x, medFiltWindow=MEDFILT_WINDOW, cutoffHz=CUTOFF_HZ):  # conditions and normalizes data
    y=filterData(x, medFiltWindow,cutoffHz)
    # compute gradients
    dy = np.gradient(y, edge_order=GRADIENT_ORDER)
    ddy = np.gradient(dy, edge_order=GRADIENT_ORDER)
    return y, dy, ddy

def filterData(x, medFiltWindow=MEDFILT_WINDOW, cutoffHz=CUTOFF_HZ):
    b, a = butter(1, cutoffHz / (FS / 2), btype='low')  # 1st order Butterworth low-pass
    # median filter outliers
    y = medfilt(x, medFiltWindow)
    # lowpass filter x
    zi = lfilter_zi(b, a)
    y, _ = lfilter(b, a, y, axis=0, zi=zi * x[0])
    return y

def norm(x):
    m = np.mean(x)
    s = np.std(x)
    y = (x - m) / s
    return y


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


def normalize(dat, mean, std):
    rep = int(dat.shape[-1] / mean.shape[0])  # there is 1 mean for each input sensor value, repeat it for each element in sequence in data
    mean = np.tile(mean, rep)
    std = np.tile(std, rep)
    dat = (dat - mean) / std
    return dat


def unnormalize(dat, mean, std):
    rep = int(dat.shape[-1] / mean.shape[0])  # results in cw_len to properly tile for applying to dat.
    # there is 1 mean for each input sensor value, repeat it for each element in sequence in data
    mean = np.tile(mean, rep)
    std = np.tile(std, rep)
    dat = dat * std + mean
    return dat


def computeNormalization(dat: numpy.array):
    '''
    Computes the special normalization of our data
    Args:
        dat: numpy array of data arranged as [sample, sensor/control vaue]

    Returns:
        mean and std, each one is a vector of values
    '''
    # Collect desired prediction
    # All terms are weighted equally by the loss function (either L1 or L2),
    # so we need to make sure that we don't have values here that are too different in range
    # Since the derivatives are computed on normalized data, but divided by the delta time in ms,
    # we need to normalize the derivatives too. (We include the delta time to make more physical units of time
    # so that the values don't change with different sample rate.)
    m = np.mean(dat, 0)
    s = np.std(dat, 0)
    m[0] = m[1] = 0
    s[0] = s[1] = 1  # handle some things specially, like sin/cos that should not be touched
    return m, s


def load_data(filepath, cw_plen, cw_flen, pw_len, pw_off, seq_len, stride=1, med_filt=MEDFILT_WINDOW, cutoff_hz=CUTOFF_HZ):
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
        medfilt: median filter window, 0 to disable

    Returns:
        Unnormalized numpy arrays
        input_data:  indexed by [sample, sequence, # sensor inputs * cw_len]
        input dict:  dictionary of input data names with key index into sensor index value
        targets:  indexed by [sample, sequence, # output sensor values * pw_len]
        target dict:  dictionary of target data names with key index into sensor index value
        actual: raw input data indexed by [sample,sensor]
        actual dict:  dictionary of actual inputs with key index into sensor index value
        mean_features, std_features, mean_targets_data, std_targets_data: the means and stds of training and raw_targets data.
            These are vectors with length # of sensorss

        The last dimension of input_data and targets is organized by
         all sensors for sample0, all sensors for sample1, .....all sensors for sampleN, where N is the prediction length
    '''

   # Load dataframe
    print('loading data from '+str(filepath))
    df = pd.read_csv(filepath)

    # time,deltaTimeMs,angle,position,angletargets,angleErr,positiontargets,positionErr,angleCmd,positionCmd,motorCmd,actualMotorCmd
    # 172.2985134124756,4.787921905517578,1699,-418,3129,-1428.0,0,-418.0,573146.4030813494,-8360.0,7055,0


    print('processing data to generate sequences')
    time = df.time.to_numpy()
    deltaTimeMs = df.deltaTimeMs.to_numpy()

    # Get Raw Data
    angle = df.angleErr.to_numpy()  # simplify the pole angle to just angle error, so it is zero centered around presumed vertical position
    # desired vertical angle might be incorrectly set, resulting in large position offset that balances position and angle control
    angle = angle * RAD_PER_ANGLE_ADC  # angle has range -pi to +pi, centered very closely around 0 during balancing
    position = df.position.to_numpy() / float(POSITION_LIMIT)  # leave cart position as absolute position since the raw_targets position might change with time and the values
    # for control might depend on position of cart
    # position has range < +/-1

    angle=filterData(angle, med_filt, cutoff_hz)
    position=filterData(position, med_filt, cutoff_hz)

    # project angle onto x and y since angle is a rotational variable with 2pi cut that we cannot fit properly and does not represent gravity and lateral acceleration well.
    sinAngle = np.sin(angle)
    cosAngle = np.cos(angle)
    # # compute temporal derivatives from state data
    # averageDeltaTMs = deltaTimeMs.mean()  # TODO this is approximate derivative since sample rate varied a bit around 5ms
    # TODO consider using a better controlled derivative that enforces e.g. total variation constraint
    actualMotorCmd = df.actualMotorCmd.to_numpy()  # zero-centered motor speed command

    # Derive Other Data
    sinAngle, dSinAngle, ddSinAngle = filterAndGradients(sinAngle, med_filt, cutoff_hz)
    cosAngle,dCosAngle, ddCosAngle=filterAndGradients(cosAngle, med_filt, cutoff_hz)
    position,dPosition, ddPosition=filterAndGradients(position, med_filt, cutoff_hz)

    # Features (Train Data)
    raw_features = []
    # raw_features.append(angle)
    # raw_features.append(dAngle)
    # raw_features.append(ddAngle)
    raw_features.append(sinAngle)
    raw_features.append(cosAngle)
    raw_features.append(dSinAngle)
    raw_features.append(dSinAngle)
    raw_features.append(ddSinAngle)
    raw_features.append(ddCosAngle)
    raw_features.append(position)
    raw_features.append(dPosition)
    raw_features.append(ddPosition)
    raw_features.append(actualMotorCmd)
    raw_features = np.vstack(raw_features).transpose()  # raw_features indexed by [sample, input sensor/control]
    features_dict={ 'sinAngle':0,'cosAngle':1,'dSinAngle':2,'dCosAngle':3,'ddSinAngle':4,'ddCosAngle':5,'position':6,'dPosition':7,'ddPosition':8,'actualMotorCmd': 9}

    # targetss (Label Data)
    raw_targets = []
    raw_targets.append(sinAngle)
    raw_targets.append(cosAngle)
    raw_targets.append(dSinAngle)
    raw_targets.append(dCosAngle)
    # raw_targets.append(angle)
    # raw_targets.append(dAngle)
    raw_targets.append(position)
    raw_targets.append(dPosition)
    raw_targets = np.vstack(raw_targets).transpose()  # raw_targets indexed by [sample, sensor]
    targets_dict={ 'sinAngle':0,'cosAngle':1,'dSinAngle':2,'dCosAngle':3,'position':4,'dPosition':5}

    # Actual Data for Plotting
    raw_actual = []
    raw_actual.append(angle)
    raw_actual.append(position)
    raw_actual.append(actualMotorCmd)
    actual = np.vstack(raw_actual).transpose()  # raw_actual data indexed by [sample, sensor]
    actual_dict = {'angle': 0, 'position': 1, 'actualMotorCmd': 2}

    # compute normalization of data now
    mean_features, std_features = computeNormalization(raw_features)
    mean_targets, std_targets = computeNormalization(raw_targets)

    # Split train data
    numSamples = raw_features.shape[0]

    features = []
    targets = []
    # Iterate from the current timestep to the last timestep that gives the last prediction window
    for i in range(cw_plen, numSamples - max(cw_flen, pw_len + pw_off) - seq_len):
        if i % stride is not 0:
            continue
        raw_feature_seq = []
        raw_target_seq = []
        for t in range(0, seq_len):
            raw_feature_seq.append(raw_features[i + t - cw_plen:i + t + cw_flen + 1, :].ravel())
            # ravel() makes 1d array for sensor values and prediction window length
            # order is row major, with last index changing fastest
            # since last index is sensor, the order is [sensor1t0, sensor2t0, ... sensorNt0, sensor0t1, sensor2t1 ....]
            raw_target_seq.append(raw_targets[i + t + pw_off:i + t + pw_off + pw_len, :].ravel())
        raw_feature_seq = np.stack(raw_feature_seq, axis=0)
        raw_target_seq = np.stack(raw_target_seq, axis=0)
        features.append(raw_feature_seq)
        targets.append(raw_target_seq)
    features = np.stack(features, axis=0)  # indexed by [sample, sequence, # sensor inputs * cw_len]
    targets = np.stack(targets, axis=0)  # [sample, sequence, # output sensor values * pw_len]

    return features, features_dict, targets, targets_dict, actual, actual_dict, mean_features, std_features, mean_targets, std_targets
