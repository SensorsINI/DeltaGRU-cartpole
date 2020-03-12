import sys
import collections
import argparse
from modules import models as models
import time
import torch.utils.data.dataloader
import numpy as np
import random as rnd
from modules.util import quantizeTensor, print_commandline, load_normalization
from modules.data import load_data, Dataset, RAD_PER_ANGLE_ADC
import matplotlib.pyplot as plt
import main
from torch.utils import data
# import matplotlib.pylab as pylab

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    parser.add_argument('--train_file', default=main.TRAIN_FILE_DEFAULT, type=str,help='(ignored) Training dataset file')
    parser.add_argument('--val_file', default=main.VAL_FILE_DEFAULT, type=str,help='(ignored) Validation dataset file')
    parser.add_argument('--test_file', default=main.TEST_FILE_DEFAULT, type=str,help='Testing dataset file')
    parser.add_argument('--seed', default=1, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--cw_plen', default=5, type=int, help='Number of previous timesteps in the context window, leads to initial latency')
    parser.add_argument('--cw_flen', default=0, type=int, help='Number of future timesteps in the context window, leads to consistent latency')
    parser.add_argument('--pw_len', default=20, type=int, help='Number of future timesteps in the prediction window')
    parser.add_argument('--pw_off', default=1, type=int, help='Offset in #timesteps of the prediction window w.r.t the current timestep')
    parser.add_argument('--pw_idx', default=1, type=int, help='Index of timestep in the prediction window to show in plots')
    parser.add_argument('--seq_len', default=100, type=int, help='Sequence Length')
    parser.add_argument('--start_test_tstep', default=0, type=int, help='Start position of the plot')
    parser.add_argument('--num_test_tstep', default=4000, type=int, help='Number of timesteps in the plot')
    parser.add_argument('--rnn_type', default='GRU', help='Mode 0 - Pretrain on GRU; Mode 1 - Retrain on GRU; Mode 2 - Retrain on DeltaGRU')
    parser.add_argument('--num_rnn_layers', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn_hid_size', default=128, type=int, help='RNN Hidden layer size')
    parser.add_argument('--qa', default=0, type=int, help='Whether quantize the network activations')
    parser.add_argument('--qw', default=0, type=int, help='Whether quantize the network weights')
    parser.add_argument('--aqi', default=8, type=int, help='Number of integer bits before decimal point for activation')
    parser.add_argument('--aqf', default=8, type=int, help='Number of integer bits after decimal point for activation')
    parser.add_argument('--wqi', default=8, type=int, help='Number of integer bits before decimal point for weight')
    parser.add_argument('--wqf', default=8, type=int, help='Number of integer bits after decimal point for weight')
    parser.add_argument('--th_x', default=64/256, type=float, help='Delta threshold for inputs')
    parser.add_argument('--th_h', default=64/256, type=float, help='Delta threshold for hidden states')
    # below ignored, added to allow running with same args as training
    parser.add_argument('--batch_size', default=64, type=int, help='(ignored) Batch size. How many samples to run forward in parallel before each weight update.')
    parser.add_argument('--num_epochs', default=5, type=int, help='(ignored) Number of epochs to train for.')
    parser.add_argument('--mode', default=1, type=int, help='(ignored) Mode 0 - Pretrain on GRU; Mode 1 - Retrain on GRU; Mode 2 - Retrain on DeltaGRU')
    parser.add_argument('--lr', default=5e-4, type=float, help='(ignored) Learning rate')  # 5e-4
    args = parser.parse_args()

    # print command line (maybe to use in a script)
    print_commandline(parser)

    # Set seeds
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Hyperparameters
    cw_plen = args.cw_plen  # Length of history in timesteps used to train the network
    cw_flen = args.cw_flen  # Length of future in timesteps to predict
    pw_len = args.pw_len  # Offset of future in timesteps to predict
    pw_off = args.pw_off  # Length of future in timesteps to predict
    pw_idx = args.pw_idx  # Index of timestep in the prediction window
    seq_len = args.seq_len  # Sequence length
    test_file=args.test_file

    # Plot Settings
    start_test_tstep = args.start_test_tstep
    num_test_tstep = args.num_test_tstep

    # Whether test retrain model
    retrain = 0


    # Network Dimension
    rnn_hid_size = args.rnn_hid_size
    rnn_type = 'GRU'
    fc_hid_size = rnn_hid_size
    num_rnn_layers = args.num_rnn_layers
    rnn_type = args.rnn_type
    th_x = args.th_x
    th_h = args.th_h
    aqi = args.aqi
    aqf = args.aqf
    wqi = args.wqi
    wqf = args.wqf

    # Save and Log
    str_target_variable = 'cart-pole'
    str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size) + 'H-'
    str_windows = str(cw_plen) + 'CWP-' + str(cw_flen) + 'CWF-' + str(pw_len) + 'PWL-' + str(pw_off) + 'PWO-'
    filename = str_net_arch + str(rnn_type) + '-' + str_windows + str_target_variable
    pretrain_model_path = str_net_arch + 'GRU'
    logpath = './log/' + filename + '.csv'
    savepath = './save/' + filename + '.pt'
    plotpath = './log/' + filename + '.svg'

    ########################################################
    # Create Dataset
    ########################################################
    _, test_data, test_labels = load_data(test_file, cw_plen, cw_flen, pw_len, pw_off, seq_len)

    # Convert data to PyTorch Tensors
    test_data = torch.Tensor(test_data).float()  # the raw sensor and control data
    test_labels = torch.Tensor(test_labels).float() # what we want to predict (the sensor data into the future)

    # Get Mean and Std of Train Data78/-
    mean_train_data, std_train_data=load_normalization(savepath) # we need to unnormalize the predictions to get the predictions in input units
    # mean_train_data = torch.mean(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)
    # std_train_data = torch.std(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)

    # Get number of classes
    rnn_output_size = test_labels.size(-1)
    print("\n")


    print('###################################################################################\n\r'
          '# Dataset\n\r'
          '###################################################################################')
    print("Dim: (num_sample, look_back_len, feat_size)")
    # print("Train data size:  ", train_data.size())
    # print("Train label size: ", train_labels.size())
    print("Test data size:         ", test_data.size())
    print("Test prediction size:   ", test_labels.size())
    print("\n\r")

    ########################################################
    # Define Network
    ########################################################
    print("Loading network...")

    # Network Dimension
    rnn_inp_size = test_data.size(-1)
    rnn_output_size = test_labels.size(-1)
    print("RNN input size:        ", rnn_inp_size)
    print("RNN output size:       ", rnn_output_size)

    # Instantiate Model
    net = models.Model(inp_size=rnn_inp_size,
                       cla_type=rnn_type,
                       cla_size=rnn_hid_size,
                       cla_layers=num_rnn_layers,
                       num_classes=rnn_output_size,
                       th_x=th_x,
                       th_h=th_h,
                       eval_sparsity=0,
                       quantize_act=1,
                       cuda=1)

    ########################################################
    # Initialize Parameters
    ########################################################
    pre_trained_model = torch.load(savepath)

    pre_trained_model = list(pre_trained_model.items())
    new_state_dict = collections.OrderedDict()
    count = 0
    num_param_key = len(pre_trained_model)
    for key, value in net.state_dict().items():
        if count >= num_param_key:
            break
        layer_name, weights = pre_trained_model[count]
        new_state_dict[key] = weights
        print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
        count += 1
    net.load_state_dict(new_state_dict)

    # Move network to GPU
    net = net.cuda()

    ########################################################
    ########################################################

    print("Starting inference...")

    # Timer
    start_time = time.time()

    ########################################################################
    # Evaluation
    ########################################################################
    test_data = test_data[start_test_tstep:start_test_tstep + num_test_tstep, 0, :].unsqueeze(1) # raw data, unsqueeze keeps as 3d
    test_data = test_data.transpose(0, 1) # put seq len as first, sample as 2nd, to normalization
    test_data_norm = (test_data - mean_train_data) / std_train_data # to input to RNN for prediction

    # Get corresponding actual series
    test_actual = test_labels[start_test_tstep:start_test_tstep + num_test_tstep, :, :].squeeze().numpy() # the actual output we will compare with

    # Run trained network
    net = net.eval() # set to eval mode
    test_sample_norm = test_data_norm.cuda()
    # test_sample_norm = quantizeTensor(test_sample_norm, aqi, aqf, 1) # TODO add check for quantization
    test_sample_norm = test_sample_norm.transpose(0, 1) # flip for input to RNN
    pred_series, pred_point, _ = net(test_sample_norm) # run samples through RNN, results in

    print('###################################################################################\n\r'
          '# Evaluation Information\n\r'
          '###################################################################################')

    print("Test Sample Size:      ", test_sample_norm.size())
    print("Test Output Size:      ", pred_series.size())
    print("Test Real Data Size:  ", test_actual.shape)

    ########################################################################
    # Plot Test Results
    ########################################################################
    # params = {'axes.titlesize': 30,
    #           'legend.fontsize': 'x-large',
    #           'figure.figsize': (15, 5),
    #          'axes.labelsize': 'x-large',
    #          'axes.titlesize':'x-large',
    #          'xtick.labelsize':'x-large',
    #          'ytick.labelsize':'x-large'}
    # pylab.rcParams.update(params)

    pred_series = pred_series.squeeze().cpu().detach().numpy() # [sample, state]

    # Select Plot Range
    y_actual = test_labels[start_test_tstep:start_test_tstep + num_test_tstep, 0, :]

    # Draw a plot of RNN input and output
    # sample_rate = 200  # 200 Hz
    t_actual = np.arange(0, num_test_tstep)
    t_pred = np.arange(pw_idx, pw_idx + num_test_tstep)

    # compute angle from sin and cos
    angle_actual = test_data[0,:,0] # should already be in radians [1,sample,pw_len*sensors]
    angle_pred=np.arctan2(pred_series[:,0],pred_series[:,1]) # sin, cos to rad

    pos_actual=test_actual[0,:,7]
    pos_pred=pred_series[:,7]  # TODO must be wrong, also needs to be unnormalized

    # Plot angle error
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    # ax3.subplots_adjust(top=0.8)
    # ax3.title.set_text('torque_desired_ankle')
    # ax3.set_ylabel("$\\tau_{da} (kg\cdot m^{2}\cdot s^{-2})$", fontsize=24)
    ax1.set_title('(a)', fontsize=24)
    ax1.set_ylabel("angle err (rad)", fontsize=24)
    ax1.set_xlabel('Time (samples, 200/s)', fontsize=24)
    ax1.plot(t_actual, angle_actual, 'k.', markersize=12, label='Ground Truth')
    ax1.plot(t_pred, angle_pred, 'r.', markersize=3, label='RNN')
    # ax1.plot(x_actual, test_actual[:, 0, pw_idx*2], 'k.', markersize=12, label='Ground Truth')
    # ax1.plot(x_pred, pred_series[0, :, pw_idx*2], 'r.', markersize=3, label='RNN')
    ax1.tick_params(axis='both', which='major', labelsize=20)
    # ax3.set_yticks(np.arange(0, 36, 5))
    ax1.legend(fontsize=18)

    # Plot position error
    ax2.set_title('(b)', fontsize=24)
    ax2.set_ylabel("position err (enc)", fontsize=24)
    ax2.set_xlabel('Time', fontsize=24)
    ax2.plot(t_actual,pos_actual , 'k.', markersize=12, label='Ground Truth')
    ax2.plot(t_pred, pos_pred, 'r.', markersize=3, label='RNN')
    ax2.tick_params(axis='both', which='major', labelsize=20)
    # ax2.legend(fontsize=18)
    # ax2.set_yticks(np.arange(-20, 120, 20))

    fig1.tight_layout(pad=1)
    fig1.savefig(plotpath, format='svg', bbox_inches='tight')
    plt.show()