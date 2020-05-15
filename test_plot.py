import collections
from modules import models as models, parseArgs
import time
import torch.utils.data.dataloader
import numpy as np
import random as rnd
from modules.util import load_normalization
from modules.data import load_data, normalize, unnormalize, computeNormalization
import matplotlib.pyplot as plt
import warnings

# import matplotlib.pylab as pylab

if __name__ == '__main__':
    args = parseArgs.args()

    # Set seeds
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Hyperparameters
    cw_plen = args.cw_plen  # Context window length of history in timesteps used to train (and run) the network
    cw_flen = args.cw_flen  # Future length in timesteps to predict
    pw_len = args.pw_len  # Offset of future in timesteps to predict
    pw_off = args.pw_off  # Length of future in timesteps to predict
    pw_idx = args.pw_idx  # Index of timestep in the prediction window
    seq_len = args.seq_len  # Sequence length
    test_file = args.test_file

    # Plot Settings
    # start_test_tstep = args.start_test_tstep
    # num_test_tstep = args.num_test_tstep
    plot_len = args.plot_len  # Number of timesteps in the plot window

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
    test_features, test_dict, test_targets,target_dict, test_actual, actual_dict, _, _, _,_ = load_data(test_file, cw_plen, cw_flen, pw_len, pw_off, seq_len, args.stride,args.med_filt, args.cutoff_hz, test_plot = True)
    #test_features = test_features[:, 0, :]  # Get a continuous time series
    # test_features: input sensor and control signals
    # test_targets: what we want to predict (the sensor data into the future)
    # both are torch tensors

    # Get Mean and Std of Train Data, to use normalize test data and unnormalize it for plotting
    mean_train_features, std_train_features, mean_train_targets, std_train_targets \
        = load_normalization(savepath)  # we need to unnormalize the predictions to get the predictions in input units
    test_features_norm = normalize(test_features, mean_train_features, std_train_features)
    test_targets_norm = normalize(test_targets, mean_train_targets, std_train_targets)

    print('# Dataset')
    print("Dim: (num_sample, look_back_len, feat_size)")
    print("Test data size:         ", test_features_norm.shape)
    print("Test prediction size:   ", test_targets_norm.shape)
    print("\n\r")

    ########################################################
    # Define Network
    ########################################################
    print("Loading network...")

    # Network Dimension
    rnn_inp_size = test_features_norm.shape[-1]
    rnn_output_size = test_targets_norm.shape[-1]
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
                       cuda=args.cuda)


    ########################################################
    # Initialize Parameters
    ########################################################
    print("Loading Model: ", savepath)
    if args.cuda == 1:
        pre_trained_model = torch.load(savepath)
    else:
        pre_trained_model = torch.load(savepath, map_location=torch.device('cpu'))
    

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
    # if args.cuda:
    #     net = net.cuda()

    ########################################################
    ########################################################

    print("Starting inference...")

    ########################################################################
    # actual_data = test_features[:, 0, :]  # Prepare actual data
    # test_features = test_features.transpose(0, 1) # put seq len as first, sample as 2nd, to normalize
    # test_features_norm = (test_features - mean_train_features) / std_train_features # to input to RNN for prediction
    # # test_sample_norm = quantizeTensor(test_sample_norm, aqi, aqf, 1) # TODO add check for quantization
    test_sample_norm = torch.from_numpy(test_features_norm).float()  # Convert Numpy to PyTorch
    test_sample_norm = test_sample_norm.unsqueeze(1)  # Convert Numpy to PyTorch
    # test_sample_norm = test_sample_norm.transpose(0, 1)  # flip for input to RNN
    # if args.cuda:
    #     test_sample_norm = test_sample_norm.cuda()  # move data to cuda

    net = net.eval()  # set to eval mode
    y_pred, _, _ = net(test_sample_norm)  # run samples through RNN, results in
    y_pred = y_pred.squeeze().cpu().detach().numpy()  # Convert tensor to numpy mat [sample, state]
    y_pred = unnormalize(y_pred, mean_train_targets, std_train_targets)  # Unnormalize network prediction

    # print('###################################################################################\n\r'
    #       '# Evaluation Information\n\r'
    #       '###################################################################################')
    #
    # print("Test Sample Size:  ", test_sample_norm.size())
    # print("Test Output Size:  ", y_pred.shape)
    # print("Actual Data Size:  ", actual_data.shape)
    # print("Test Label Size:   ", test_targets.shape)

    ########################################################################
    # Draw a plot of RNN input and output
    ########################################################################
    # Get Number of Total Timesteps
    num_test_tstep = test_sample_norm.size(0)

    # Get Ground Truth
    t_start = cw_plen  # Put the initial timestep at the first timestep after the first possible context window
    t_start = 0
    ts_actual = np.arange(t_start, t_start + num_test_tstep)  # timesteps to show actual input data
    angle_actual = test_actual[ts_actual, actual_dict['angle']].squeeze()  # actual original data of input angle
    position_actual = test_actual[ts_actual, actual_dict['position']].squeeze()  # actual input cart position, normalized to table size by POSITION_LIMIT
    motor_actual = test_actual[ts_actual, actual_dict['actualMotorCmd']].squeeze()  # original input motor cmd
    positionTarget = test_actual[ts_actual, actual_dict['positionTarget']].squeeze()  # original input motor cmd

    # Get Predictions
    ts_pred = np.arange(t_start + pw_off,t_start + pw_off + num_test_tstep)  # prediction timesteps, not quite the same since there is offset and window
    y_pred = np.reshape(y_pred, (num_test_tstep, pw_len,-1))  #  doesn't use the ts_pred because shape adds the feature dimension (timestep, pw_len, feat)
    sin_pred = np.squeeze(y_pred[t_start:t_start + num_test_tstep, 0, target_dict['sinAngle']])
    cos_pred = np.squeeze(y_pred[t_start:t_start + num_test_tstep, 0, target_dict['cosAngle']])
    angle_pred = np.arctan2(sin_pred, cos_pred)  # compute angle from sin and cos
    position_pred = np.squeeze(y_pred[t_start:t_start + num_test_tstep, 0, target_dict['position']])  # prediction of normalized position
#%%
    # Plot angle error
    fig1, axs = plt.subplots(4, 1, figsize=(14, 8), sharex=True) # share x axis so zoom zooms all plots
    # axs[0].set_title('(a)', fontsize=24)
    axs[0].set_ylabel("angle err (rad)", fontsize=18)
    axs[0].plot(ts_actual, angle_actual, 'k', markersize=12, label='Ground Truth')
    axs[0].plot(ts_pred, angle_pred, 'r', markersize=3, label='RNN')
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[0].legend(fontsize=14)

    # Plot position
    # axs[1].set_title('(b)', fontsize=24)
    axs[1].set_ylabel("position (norm)", fontsize=18)
    # axs[1].set_xlabel('Time', fontsize=18)
    axs[1].plot(ts_actual, position_actual, 'k', markersize=12, label='Ground Truth')
    axs[1].plot(ts_pred, position_pred, 'r', markersize=3, label='RNN')
    # print("t_pred size:      ", t_pred.shape)
    # print("pred_series size: ", position_actual.shape)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].legend(fontsize=14)

    # plot motor input command
    # axs[2].set_title('(b)', fontsize=24)
    axs[2].set_ylabel("motor (norm)", fontsize=18)
    axs[2].plot(ts_actual, motor_actual, 'k', markersize=12, label='motor')
    axs[2].tick_params(axis='both', which='major', labelsize=16)

    axs[3].set_ylabel("position target", fontsize=18)
    axs[3].set_xlabel('Time (samples, 200/s)', fontsize=18)
    axs[3].plot(ts_actual, positionTarget, 'k')
    axs[3].tick_params(axis='both', which='major', labelsize=16)


#%%

    ########################################################################
    # Slider
    ########################################################################
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    # Initialize data for slider plots
    # t = np.arange(0.0, 1.0, 0.001)
    t_start = 0
    t_curr = cw_plen  # Put the initial timestep at the first timestep after the first possible context window
    ts_sliderplot = np.arange(t_start, t_start + plot_len)
    ts_context = np.arange(t_start, t_start + cw_plen)
    ts_pred = np.arange(t_curr + pw_off, t_curr + pw_off + pw_len)

    angle_actual = test_actual[ts_sliderplot, actual_dict['angle']].squeeze()
    angle_context = test_actual[ts_context, actual_dict['angle']].squeeze()
    # indexes of y_pred are [sample, predition_window, sensor_output], e.g. [9000, 50, 6]
    # at this point y_pred (RNN output) is already unnormalized
    sin_pred = np.squeeze(y_pred[t_start, :, target_dict['sinAngle']]) # for slider, prediction is all pw_len (e.g. 50) predictions at t_start for sin
    cos_pred = np.squeeze(y_pred[t_start, :, target_dict['sinAngle']])
    angle_pred = np.arctan2(sin_pred, cos_pred)
    position_actual_sliderplot = position_actual[ts_sliderplot]
    position_context = position_actual[ts_context]
    position_pred = y_pred[t_start, :, target_dict['position']]
    motor_actual = test_actual[ts_sliderplot, actual_dict['actualMotorCmd']].squeeze()
    positionTarget = test_actual[ts_sliderplot, actual_dict['positionTarget']].squeeze()

    # Draw Plots, from top position, angle, motor
    fig, axs = plt.subplots(4, 1,figsize=(14, 10), sharex=True)
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.3, top=0.9)
    # pole angle
    plot0_actual, =     axs[0].plot(ts_sliderplot, angle_actual, 'k', lw=3, label='Ground Truth')
    plot0_context, =    axs[0].plot(ts_context, angle_context, 'g--', lw=6, label='Context')
    plot0_pred, =       axs[0].plot(ts_pred, angle_pred, 'r-', lw=4, label='Prediction')
    # cart position
    plot1_actual, =     axs[1].plot(ts_sliderplot, position_actual_sliderplot, 'k', lw=3, label='Ground Truth')
    plot1_context, =    axs[1].plot(ts_context, position_context, 'g--', lw=6, label='Context')
    plot1_pred, =       axs[1].plot(ts_pred, position_pred, 'r-', lw=4, label='Prediction')
    # motor
    plot2_actual, =     axs[2].plot(ts_sliderplot, motor_actual, 'k', lw=1, label='Motor')
    # position target (from  user or dance program)
    plot3_actual, =     axs[3].plot(ts_sliderplot, positionTarget, 'k', lw=1, label='Motor')

    axs[0].set_ylabel("Ang. Error (rad)", fontsize=12)
    # axs[0].set_ylim((-0.1, 0.1)) # 0.1 is 5.7 deg
    axs[0].legend(fontsize=12)

    axs[1].set_ylabel("Position (norm)", fontsize=12)
    # axs[1].set_ylim((-2,2))
    axs[1].legend(fontsize=12)
    # axs[1].margins(x=0)

    axs[2].set_ylabel("motor (PWM)", fontsize=12)
    axs[2].legend(fontsize=12)

    axs[3].set_ylabel("position target", fontsize=12)
    axs[3].set_xlabel('Timestep (200 Hz)', fontsize=12)

    axcolor = 'lightgoldenrodyellow'
    axtstep = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)

    # Sliders
    a0 = 5
    f0 = 3
    delta_step = .2
    ststep = Slider(axtstep, 'Timestep', cw_plen, num_test_tstep - pw_len - pw_off - plot_len, valinit=f0,
                    valstep=delta_step)


    def update(val):
        t_curr = int(ststep.val)  # Put the initial timestep at the first timestep after the first possible context window
        t_start = t_curr - cw_plen
        ts_sliderplot = np.arange(t_start, t_start + plot_len)
        ts_context = np.arange(t_start, t_start + cw_plen)
        ts_pred = np.arange(t_curr + pw_off, t_curr + pw_off + pw_len)

        angle_actual = test_actual[ts_sliderplot, actual_dict['angle']].squeeze()
        angle_context = test_actual[ts_context, actual_dict['angle']].squeeze()
        sin_pred = np.squeeze(y_pred[t_start, :, target_dict['sinAngle']])
        cos_pred = np.squeeze(y_pred[t_start, :, target_dict['cosAngle']])
        angle_pred = np.arctan2(sin_pred, cos_pred)
        position_act = position_actual[ts_sliderplot]
        position_context = position_actual[ts_context]
        position_pred = y_pred[t_start, :, target_dict['position']]
        motor_actual = test_actual[ts_sliderplot, actual_dict['actualMotorCmd']].squeeze()
        positionTarget = test_actual[ts_sliderplot, actual_dict['positionTarget']].squeeze()

        plot0_actual.set_xdata(ts_sliderplot)
        plot0_context.set_xdata(ts_context)
        plot0_pred.set_xdata(ts_pred)
        plot0_actual.set_ydata(angle_actual)
        plot0_context.set_ydata(angle_context)
        plot0_pred.set_ydata(angle_pred)

        plot1_actual.set_xdata(ts_sliderplot)
        plot1_context.set_xdata(ts_context)
        plot1_pred.set_xdata(ts_pred)
        plot1_actual.set_ydata(position_act)
        plot1_context.set_ydata(position_context)
        plot1_pred.set_ydata(position_pred)

        plot2_actual.set_xdata(ts_sliderplot)
        plot2_actual.set_ydata(motor_actual)

        plot3_actual.set_xdata(ts_sliderplot)
        plot3_actual.set_ydata(positionTarget)

        # https: // stackoverflow.com / questions / 10984085 / automatically - rescale - ylim - and -xlim - in -matplotlib
        for ax in axs:
            ax.set_xlim(xmin=t_start, xmax=t_start + plot_len)
            ax.relim()
            ax.autoscale_view()

    ststep.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        ststep.reset()


    button.on_clicked(reset)

    plt.show()
