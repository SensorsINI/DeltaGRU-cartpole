import collections
import argparse
from modules import models as models
import time
import torch.utils.data.dataloader
import numpy as np
import random as rnd
from modules.util import quantizeTensor
from modules.data import load_data, Dataset
import matplotlib.pyplot as plt
from torch.utils import data
# import matplotlib.pylab as pylab

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    parser.add_argument('--seed', default=1, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--cw_plen', default=20, type=int, help='Number of previous timesteps in the context window, leads to initial latency')
    parser.add_argument('--cw_flen', default=0, type=int, help='Number of future timesteps in the context window, leads to consistent latency')
    parser.add_argument('--pw_len', default=5, type=int, help='Number of future timesteps in the prediction window')
    parser.add_argument('--pw_off', default=1, type=int, help='Offset in #timesteps of the prediction window w.r.t the current timestep')
    parser.add_argument('--pw_idx', default=1, type=int, help='Index of timestep in the prediction window to show in plots')
    parser.add_argument('--seq_len', default=100, type=int, help='Sequence Length')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of epochs to train for.')
    parser.add_argument('--rnn_type', default='GRU', help='Mode 0 - Pretrain on GRU; Mode 1 - Retrain on GRU; Mode 2 - Retrain on DeltaGRU')
    parser.add_argument('--num_rnn_layers', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn_hid_size', default=512, type=int, help='RNN Hidden layer size')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')  # 5e-4
    parser.add_argument('--qa', default=0, type=int, help='Whether quantize the network activations')
    parser.add_argument('--qw', default=1, type=int, help='Whether quantize the network weights')
    parser.add_argument('--aqi', default=8, type=int, help='Number of integer bits before decimal point for activation')
    parser.add_argument('--aqf', default=8, type=int, help='Number of integer bits after decimal point for activation')
    parser.add_argument('--wqi', default=8, type=int, help='Number of integer bits before decimal point for weight')
    parser.add_argument('--wqf', default=8, type=int, help='Number of integer bits after decimal point for weight')
    parser.add_argument('--th_x', default=64/256, type=float, help='Delta threshold for inputs')
    parser.add_argument('--th_h', default=64/256, type=float, help='Delta threshold for hidden states')
    args = parser.parse_args()

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
    seq_len = args.seq_len  # Sequence length
    lr = args.lr  # Learning rate
    batch_size = args.batch_size  # Mini-batch size

    # Plot Settings
    start_time_step = 0 #2100
    num_test_sample = 4000 #600

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


    ########################################################
    # Create Dataset
    ########################################################
    _, data_1, labels_1 = load_data('data/cartpole-2020-02-27-14-14-23 pololu control plus free plus cart response.csv', cw_plen, cw_flen, pw_len, pw_off, seq_len)
    _, data_2, labels_2 = load_data('data/cartpole-2020-02-27-14-18-18 pololu PD control.csv', cw_plen, cw_flen, pw_len, pw_off, seq_len)
    _, data_3, labels_3 = load_data('data/cartpole-2020-02-21-10-12-40.csv', cw_plen, cw_flen, pw_len, pw_off, seq_len)

    train_ampro_data = data_1 #np.concatenate((data_1), axis=0) # if only one file, then don't concatenate, it kills an axis
    train_ampro_labels = labels_1 #np.concatenate((labels_1), axis=0)
    test_ampro_data = data_3
    test_ampro_labels = labels_3

    # Convert data to PyTorch Tensors
    train_data = torch.Tensor(train_ampro_data).float()
    train_labels = torch.Tensor(train_ampro_labels).float()
    test_data = torch.Tensor(test_ampro_data).float()
    test_labels = torch.Tensor(test_ampro_labels).float()

    # Get Mean and Std of Train Data78/-
    mean_train_data = torch.mean(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)
    std_train_data = torch.std(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)

    # Get number of classes
    num_classes = train_labels.size(-1)
    print("\n")


    print('###################################################################################\n\r'
          '# Dataset\n\r'
          '###################################################################################')
    print("Dim: (num_sample, look_back_len, feat_size)")
    print("Train data size:  ", train_data.size())
    print("Train label size: ", train_labels.size())
    print("Test data size:   ", test_data.size())
    print("Test label size:  ", test_labels.size())
    print("\n\r")

    ########################################################
    # Define Network
    ########################################################

    # Network Dimension
    rnn_inp_size = train_data.size(-1)
    num_classes = train_labels.size(-1)
    print("RNN Input Size:          ", rnn_inp_size)
    print("Number of Classes:       ", num_classes)

    # Instantiate Model
    net = models.Model(inp_size=rnn_inp_size,
                       cla_type=rnn_type,
                       cla_size=rnn_hid_size,
                       cla_layers=num_rnn_layers,
                       num_classes=num_classes,
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

    # Epoch loop
    print("Starting testing...")

    # Timer
    start_time = time.time()

    epoch_nz_dx = 0
    epoch_nz_dh = 0


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



    # Continuous Test
    test_data = test_data[start_time_step:start_time_step+num_test_sample, 0, :].unsqueeze(1)
    test_data = test_data.transpose(0, 1)
    test_data_norm = (test_data-mean_train_data)/std_train_data

    # Get corresponding actual series
    test_actual = test_labels[start_time_step:start_time_step+num_test_sample, :, :].squeeze().numpy()

    # Run trained network
    net = net.eval()
    test_sample_norm = test_data_norm.cuda()

    test_sample_norm = quantizeTensor(test_sample_norm, aqi, aqf, 1)
    test_sample_norm = test_sample_norm.transpose(0, 1)
    print("Test Sample Size: ", test_sample_norm.size())
    pred_series, pred_point, _ = net(test_sample_norm)
    print("Test Output Size: ", test_sample_norm.size())
    pred_point = pred_point.cpu().squeeze().detach().numpy()
    pred_series = pred_series.cpu().detach().numpy()
    print(test_actual.shape)
    print(pred_series.shape)

    # Select Plot Range
    y_actual = test_labels[start_time_step:start_time_step+num_test_sample, 0, :]
    y_pred = pred_series

    # Draw a plot of RNN input and output
    sample_rate = 200  # 200 Hz
    x_actual = np.arange(0, num_test_sample)/sample_rate
    x_pred = np.arange(pw_idx, pred_len+num_test_sample)/sample_rate

    # Plot velocity_actual_ankle
    fig1, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8))
    # ax3.subplots_adjust(top=0.8)
    # ax3.title.set_text('torque_desired_ankle')
    # ax3.set_ylabel("$\\tau_{da} (kg\cdot m^{2}\cdot s^{-2})$", fontsize=24)
    ax3.set_title('(a)', fontsize=24)
    ax3.set_ylabel("$\\tau^{d}_{pa} (kg\cdot m^{2}\cdot s^{-2})$", fontsize=24)
    ax3.set_xlabel('t (s)', fontsize=24)
    ax3.plot(x_actual, test_actual[:, 0, 0], 'k.', markersize=12, label='PD')
    ax3.plot(x_pred, pred_series[0, :, 0], 'r.', markersize=3, label='RNN')
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.set_yticks(np.arange(0, 36, 5))
    ax3.legend(fontsize=18)

    # fig1.savefig('./plot/torque_desired_ankle.svg', format='svg', bbox_inches='tight')


    # # Plot velocity_actual_knee
    # ax4 = plt.subplot(2, 1, 2)
    ax4.set_title('(b)', fontsize=24)
    ax4.set_ylabel("$\\tau^{d}_{pk} (kg\cdot m^{2}\cdot s^{-2})$", fontsize=24)
    ax4.set_xlabel('t (s)', fontsize=24)
    # ax4.plot(x_actual, test_actual[:, 0, 1], 'k.', linewidth=5, label='PD')
    # ax4.plot(x_pred, pred_series[0, :, 1], 'r.', label='RNN')
    ax4.plot(x_actual, test_actual[:, 0, 1], 'k.', markersize=12, label='PD')
    ax4.plot(x_pred, pred_series[0, :, 1], 'r.', markersize=3, label='RNN')
    ax4.tick_params(axis='both', which='major', labelsize=20)
    #ax4.plot(x_pred, test_drnn[:, 0, 1], 'gx', label='DRNN Output(Predicted)')
    # ax4.legend(fontsize=18)
    ax4.set_yticks(np.arange(-20, 120, 20))

    fig1.tight_layout(pad=1)
    fig1.savefig('./plot/eval.svg', format='svg', bbox_inches='tight')

    # # Plot control_signal_ankle
    # ax5 = plt.subplot(8, 1, 5)
    # ax5.title.set_text('control_signal_ankle')
    # ax5.set_ylabel('y')
    # ax5.set_xlabel('t (s)')
    # ax5.plot(x_actual, test_actual[:, 0, 4], 'r-', label='Input (Actual)')
    # ax5.plot(x_pred, pred_series[:, 0, 4], 'bx', label='Output(Predicted)')
    # ax5.legend()
    #
    # # Plot control_signal_knee
    # ax6 = plt.subplot(8, 1, 6)
    # ax6.title.set_text('control_signal_knee')
    # ax6.set_ylabel('y')
    # ax6.set_xlabel('t (s)')
    # ax6.plot(x_actual, test_actual[:, 0, 5], 'r.', label='Input (Actual)')
    # ax6.plot(x_pred, pred_series[:, 0, 5], 'bx', label='Output(Predicted)')
    # ax6.legend()
    #
    # # Plot torque_actual_ankle
    # ax7 = plt.subplot(8, 1, 7)
    # ax7.title.set_text('torque_actual_ankle')
    # ax7.set_ylabel('y')
    # ax7.set_xlabel('t (s)')
    # ax7.plot(x_actual, test_actual[:, 0, 6], 'r-', label='Input (Actual)')
    # ax7.plot(x_pred, pred_series[:, 0, 6], 'bx', label='Output(Predicted)')
    # ax7.legend()
    #
    # # Plot torque_actual_knee
    # ax8 = plt.subplot(8, 1, 8)
    # ax8.title.set_text('torque_actual_knee')
    # ax8.set_ylabel('y')
    # ax8.set_xlabel('t (s)')
    # ax8.plot(x_actual, test_actual[:, 0, 7], 'r-', label='Input (Actual)')
    # ax8.plot(x_pred, pred_series[:, 0, 7], 'bx', label='Output(Predicted)')
    # ax8.legend()


    plt.show()