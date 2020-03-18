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
    parser.add_argument('--cw_plen', default=100, type=int, help='Number of previous timesteps in the context window, leads to initial latency')
    parser.add_argument('--cw_flen', default=0, type=int, help='Number of future timesteps in the context window, leads to consistent latency')
    parser.add_argument('--pw_len', default=100, type=int, help='Number of future timesteps in the prediction window')
    parser.add_argument('--pw_off', default=1, type=int, help='Offset in #timesteps of the prediction window w.r.t the current timestep')
    parser.add_argument('--pw_idx', default=1, type=int, help='Index of timestep in the prediction window to show in plots')
    parser.add_argument('--seq_len', default=50, type=int, help='Sequence Length')
    parser.add_argument('--plot_len', default=500, type=int, help='Number of timesteps in the plot window')
    parser.add_argument('--rnn_type', default='GRU', help='Mode 0 - Pretrain on GRU; Mode 1 - Retrain on GRU; Mode 2 - Retrain on DeltaGRU')
    parser.add_argument('--num_rnn_layers', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn_hid_size', default=512, type=int, help='RNN Hidden layer size')
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
    # start_test_tstep = args.start_test_tstep
    # num_test_tstep = args.num_test_tstep
    plot_len = args.plot_len # Number of timesteps in the plot window

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
    print("Loading Model: ", savepath)
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

    # Timertest_actual
    start_time = time.time()

    ########################################################################
    test_data = test_data[:, 0, :].unsqueeze(1) # raw data in a time serie, unsqueeze keeps as 3d
    test_data = test_data.transpose(0, 1) # put seq len as first, sample as 2nd, to normalization
    test_data_norm = (test_data - mean_train_data) / std_train_data # to input to RNN for prediction

    # Get corresponding actual series
    test_actual = test_labels[:, 0, :].squeeze().numpy() # the actual output we will compare with

    # Run trained network
    net = net.eval() # set to eval mode
    test_sample_norm = test_data_norm.cuda()
    # test_sample_norm = quantizeTensor(test_sample_norm, aqi, aqf, 1) # TODO add check for quantization
    test_sample_norm = test_sample_norm.transpose(0, 1) # flip for input to RNN
    y_pred, _, _ = net(test_sample_norm) # run samples through RNN, results in
    y_pred = y_pred.squeeze().cpu().detach().numpy() # Convert tensor to numpy mat [sample, state]

    print('###################################################################################\n\r'
          '# Evaluation Information\n\r'
          '###################################################################################')

    print("Test Sample Size:      ", test_sample_norm.size())
    print("Test Output Size:      ", y_pred.shape)
    print("Test Real Data Size:   ", test_actual.shape)

    ########################################################################
    # Draw a plot of RNN input and output
    # sample_rate = 200  # 200 Hz
    ########################################################################
    # params = {'axes.titlesize': 30,
    #           'legend.fontsize': 'x-large',
    #           'figure.figsize': (15, 5),
    #          'axes.labelsize': 'x-large',
    #          'axes.titlesize':'x-large',
    #          'xtick.labelsize':'x-large',
    #          'ytick.labelsize':'x-large'}
    # pylab.rcParams.update(params)


    # Get Number of Total Timesteps
    num_test_tstep = test_sample_norm.size(0)

    # Get Ground Truth
    t_curr = cw_plen  # Put the initial timestep at the first timestep after the first possible context window
    t_plot = t_curr - cw_plen
    t_actual = np.arange(t_plot, t_plot + num_test_tstep)
    angle_actual = test_data[:,t_plot:t_plot + num_test_tstep,0].squeeze() # should already be in radians
    position_actual = test_data[:,t_plot:t_plot + num_test_tstep,3].squeeze() # should already be in radians

    # Get Angle Prediction
    # t_pred = np.arange(t_curr + pw_off, t_curr + pw_off + pw_len)
    t_pred = np.arange(t_curr + pw_off, t_curr + pw_off + num_test_tstep)
    y_pred = np.reshape(y_pred, (num_test_tstep, pw_len, -1))  # Reshape to add the feature dimension (timestep, pw_len, feat)
    # sin_pred = np.squeeze(y_pred[t_plot, :, 0])
    # cos_pred = np.squeeze(y_pred[t_plot, :, 1])
    # angle_pred = np.arctan2(sin_pred, cos_pred) # compute angle from sin and cos
    # sin_pred = np.squeeze(y_pred[t_plot:t_plot + plot_len, 0, 0])
    # cos_pred = np.squeeze(y_pred[t_plot:t_plot + plot_len, 0, 1])
    # angle_pred = np.arctan2(sin_pred, cos_pred) # compute angle from sin and cos
    angle_pred = np.squeeze(y_pred[t_plot:t_plot + num_test_tstep, 0, 0])

    # Get Position Prediction
    # position_pred = y_pred[t_plot, :, 2]
    position_pred = y_pred[t_plot:t_plot + num_test_tstep, 0, 1]

    # Plot angle error
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.set_title('(a)', fontsize=24)
    ax1.set_ylabel("angle err (rad)", fontsize=24)
    ax1.set_xlabel('Time (samples, 200/s)', fontsize=24)
    ax1.plot(t_actual, angle_actual, 'k.', markersize=12, label='Ground Truth')
    ax1.plot(t_pred, angle_pred, 'r.', markersize=3, label='RNN')
    ax1.tick_params(axis='both', which='major', labelsize=20)
    # ax3.set_yticks(np.arange(0, 36, 5))
    ax1.legend(fontsize=18)

    # Plot position error
    ax2.set_title('(b)', fontsize=24)
    ax2.set_ylabel("position err (enc)", fontsize=24)
    ax2.set_xlabel('Time', fontsize=24)
    ax2.plot(t_actual, position_actual, 'k.', markersize=12, label='Ground Truth')
    # print("t_pred size:      ", t_pred.shape)
    # print("pred_series size: ", position_actual.shape)
    ax2.plot(t_pred, position_pred, 'r.', markersize=3, label='RNN')
    # ax2.plot(t_actual, position_pred, 'r.', markersize=3, label='RNN')
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.legend(fontsize=18)
    # ax2.set_yticks(np.arange(-20, 120, 20))

    # fig1.tight_layout(pad=1)
    # fig1.savefig(plotpath, format='svg', bbox_inches='tight')
    # plt.show()

    ########################################################################
    # Slider
    ########################################################################
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    import math

    # Initialize Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.3, top=0.9)
    t = np.arange(0.0, 1.0, 0.001)
    t_curr = cw_plen  # Put the initial timestep at the first timestep after the first possible context window
    t_plot = t_curr - cw_plen
    t_actual = np.arange(t_plot, t_plot + plot_len)
    t_context = np.arange(t_plot, t_plot + cw_plen)
    t_pred = np.arange(t_curr + pw_off, t_curr + pw_off + pw_len)
    angle_actual = test_data[:,t_plot:t_plot + plot_len,0].squeeze() # should already be in radians
    angle_context = test_data[:,t_plot:t_plot + cw_plen,0].squeeze() # should already be in radians
    angle_pred = np.squeeze(y_pred[t_plot, :, 0])
    position_actual = test_data[:,t_plot:t_plot + plot_len,3].squeeze() # should already be in radians
    position_context = test_data[:,t_plot:t_plot + cw_plen,3].squeeze() # should already be in radians
    position_pred = y_pred[t_plot, :, 1]

    # Draw Plots
    plot1_actual, = ax1.plot(t_actual, position_actual, 'k.', lw=3, label='Ground Truth')
    plot1_context, = ax1.plot(t_context, position_context, 'g--', lw=2, label='Context')
    plot1_pred, = ax1.plot(t_pred, position_pred, 'r-', lw=2, label='Prediction')
    plot2_actual, = ax2.plot(t_actual, angle_actual, 'k.', lw=3, label='Ground Truth')
    plot2_context, = ax2.plot(t_context, angle_context, 'g--', lw=2, label='Context')
    plot2_pred, = ax2.plot(t_pred, angle_pred, 'r-', lw=2, label='Prediction')

    ax1.set_ylabel("Pos. Error (rad)", fontsize=14)
    ax1.set_yticks(np.arange(-0.1, 1.1, 0.2))
    ax1.legend(fontsize=12)
    ax2.set_ylabel("Ang. Error (rad)", fontsize=14)
    ax2.set_xlabel('Timestep (200 Hz)', fontsize=14)
    ax2.set_yticks(np.arange(-0.1, 1.1, 0.2))
    ax2.legend(fontsize=12)
    ax1.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    axtstep = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)

    # Sliders
    a0 = 5
    f0 = 3
    delta_step = 1.0
    ststep = Slider(axtstep, 'Timestep', cw_plen, num_test_tstep-pw_len-pw_off-plot_len, valinit=f0, valstep=delta_step)

    def update(val):
        t_curr = int(ststep.val)  # Put the initial timestep at the first timestep after the first possible context window
        t_plot = t_curr - cw_plen
        t_actual = np.arange(t_plot, t_plot + plot_len)
        t_context = np.arange(t_plot, t_plot + cw_plen)
        t_pred = np.arange(t_curr + pw_off, t_curr + pw_off + pw_len)
        angle_actual = test_data[:,t_plot:t_plot + plot_len,0].squeeze() # should already be in radians
        angle_context = test_data[:,t_plot:t_plot + cw_plen,0].squeeze() # should already be in radians
        angle_pred = np.squeeze(y_pred[t_plot, :, 0])
        position_actual = test_data[:,t_plot:t_plot + plot_len,3].squeeze() # should already be in radians
        position_context = test_data[:,t_plot:t_plot + cw_plen,3].squeeze() # should already be in radians
        position_pred = y_pred[t_plot, :, 1]
        plot1_actual.set_xdata(t_actual)
        plot1_context.set_xdata(t_context)
        plot1_pred.set_xdata(t_pred)
        plot1_actual.set_ydata(position_actual)
        plot1_context.set_ydata(position_context)
        plot1_pred.set_ydata(position_pred)
        plot2_actual.set_xdata(t_actual)
        plot2_context.set_xdata(t_context)
        plot2_pred.set_xdata(t_pred)
        plot2_actual.set_ydata(angle_actual)
        plot2_context.set_ydata(angle_context)
        plot2_pred.set_ydata(angle_pred)
        # ax1.set_xticks(t_actual)
        ax1.set_xlim(xmin=t_plot, xmax=t_plot + plot_len)
        ax2.set_xlim(xmin=t_plot, xmax=t_plot + plot_len)
        # ax2.set_xticks(t_actual)


    ststep.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        ststep.reset()
    button.on_clicked(reset)

    plt.show()