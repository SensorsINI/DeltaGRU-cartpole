import os
import collections
import math
import time
import torch as t
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.utils.data.dataloader
from tqdm import tqdm
import random as rnd
import numpy as np
import modules.models as models
from modules.util import quantizeTensor, timeSince, quantize_rnn, save_normalization
from modules.log import write_log
from modules.data import load_data, Dataset
from modules.deltarnn import get_temporal_sparsity
from modules import parseArgs

if __name__ == '__main__':
    args = parseArgs.args()

    # Make folders
    try:
        os.makedirs('save')
        os.makedirs('log')
    except:
        pass

    # Set seeds
    seed = args.seed
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Hyperparameters
    stride = args.stride
    cw_plen = args.cw_plen            # Length of history in timesteps used to train the network
    cw_flen = args.cw_flen            # Length of future in timesteps to predict
    pw_len = args.pw_len              # Offset of future in timesteps to predict
    pw_off = args.pw_off              # Length of future in timesteps to predict
    seq_len = args.seq_len            # Sequence length
    lr = args.lr                      # Learning rate
    batch_size = args.batch_size      # Mini-batch size
    num_epochs = args.num_epochs      # Number of epoches to train the network
    mode = args.mode
    train_file=args.train_file
    val_file=args.val_file
    test_file=args.test_file
    qa = args.qa
    qw = args.qw

    print('###################################################################################\n\r'
          '# Hyperparameters\n\r'
          '###################################################################################')
    print("mode       =  ", mode)
    print("cw_plen    =  ", cw_plen)
    print("cw_flen    =  ", cw_flen)
    print("pw_len     =  ", pw_len)
    print("pw_off     =  ", pw_off)
    print("lr         =  ", lr)
    print("batch_size =  ", batch_size)
    print("num_epochs =  ", num_epochs)

    # Network Dimension
    rnn_hid_size = args.rnn_hid_size
    if mode == 0:  # Pretrain on GRU
        rnn_type = 'GRU'
    elif mode == 1:  # Retrain on GRU
        rnn_type = 'GRU'
    elif mode == 2:  # Retrain on DeltaGRU
        rnn_type = 'DeltaGRU'

    fc_hid_size = rnn_hid_size
    num_rnn_layers = args.num_rnn_layers
    th_x = args.th_x
    th_h = args.th_h
    aqi = args.aqi
    aqf = args.aqf
    wqi = args.wqi
    wqf = args.wqf
    # Save and Log
    str_target_variable = 'cart-pole'
    save_path={}
    if mode == 0:  # Pretrain on GRU
        str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size) + 'H-'
        str_windows = str(cw_plen) + 'CWP-' + str(cw_flen) + 'CWF-' + str(pw_len) + 'PWL-' + str(pw_off) + 'PWO-'
        filename = str_net_arch + str(rnn_type) + '-' + str_windows + str_target_variable
        pretrain_model_path = str_net_arch + 'GRU'
        logpath = './log/' + filename + '.csv'
        savepath = './save/' + filename + '.pt'
    elif mode == 1:  # Retrain on GRU
        str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size) + 'H-'
        str_windows = str(cw_plen) + 'CWP-' + str(cw_flen) + 'CWF-' + str(pw_len) + 'PWL-' + str(pw_off) + 'PWO-'
        filename = str_net_arch + str(rnn_type) + '-' + str_windows + str_target_variable
        pretrain_model_path = './save/' + filename + '.pt'
        logpath = './log/' + filename + '.csv'
        savepath = './save/' + filename + '.pt'
    elif mode == 2:  # Retrain on DeltaGRU
        str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size) + 'H-'
        str_windows = str(cw_plen) + 'CWP-' + str(cw_flen) + 'CWF-' + str(pw_len) + 'PWL-' + str(pw_off) + 'PWO-'
        filename = str_net_arch + str(rnn_type) + '-' + str_windows + str_target_variable
        pretrain_model_path = './save/' + str_net_arch + 'GRU' + '-' + str_windows + str_target_variable + '.pt' #TODO probably a bug in this filename, has extra parts
        logpath = './log/' + filename + '_' + str(th_x) + '.csv'
        savepath = './save/' + filename + '_' + str(th_x) + '.pt'


    ########################################################
    # Create Dataset
    ########################################################
    train_data, train_labels, train_mean, train_std, label_mean, label_std = load_data(train_file, cw_plen, cw_flen, pw_len, pw_off, seq_len)
    dev_data, dev_labels, _, _, _, _ = load_data(val_file, cw_plen, cw_flen, pw_len, pw_off, seq_len)
    test_data, test_labels, _, _, _, _ = load_data(test_file, cw_plen, cw_flen, pw_len, pw_off, seq_len)

    save_normalization(savepath,train_mean,train_std)

      # Get number of classes
    num_classes = train_labels.size(-1)

    # Convert Dev and Test data into single batch form
    # dev_data_norm = dev_data_norm[:, 0, :].unsqueeze(1)
    # dev_data_norm = dev_data_norm.cuda()
    # dev_ampro_labels = dev_ampro_labels[:, 0, :].unsqueeze(1)
    # dev_ampro_labels = dev_ampro_labels.cuda()
    # test_data_norm = test_data_norm[:, 0, :].unsqueeze(1)
    # test_data_norm = test_data_norm.cuda()
    # test_data_norm = test_data_norm[:, 0, :].unsqueeze(1)
    # test_ampro_labels = test_data_norm.cuda()
    # print("Train data  dimension: ", train_data_norm.shape)
    # print("Train label dimension: ", train_ampro_labels.shape)
    # print("Dev   data  dimension: ", dev_data_norm.shape)
    # print("Dev   label dimension: ", dev_ampro_labels.shape)
    # print("Test  data  dimension: ", test_data_norm.shape)
    # print("Test  label dimension: ", test_ampro_labels.shape)
    print("\n")

    # Create PyTorch Dataset
    train_set = Dataset(train_data, train_labels, mode)
    dev_set = Dataset(dev_data, dev_labels, mode)
    test_set = Dataset(test_data, test_labels, mode)

    # Create PyTorch dataloaders for train and dev set
    train_generator = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    dev_generator = data.DataLoader(dataset=dev_set, batch_size=512, shuffle=True)
    test_generator = data.DataLoader(dataset=test_set, batch_size=512, shuffle=True)

    print('###################################################################################\n\r'
          '# Dataset\n\r'
          '###################################################################################')
    print("Dim: (num_sample, look_back_len, feat_size)")
    print("Train data size:  ", train_data.size())
    print("Train label size: ", train_labels.size())
    print("Test data size:   ", dev_data.size())
    print("Test label size:  ", dev_labels.size())
    print("\n\r")
    # Data Range
    angle_min = 0
    angle_max = 2 * math.pi
    position_min = -655
    position_max = 1644

    print('###################################################################################\n\r'
          '# Network\n\r'
          '###################################################################################')
    # Network Dimension
    rnn_inp_size = train_data.size(-1)
    print("rnn_inp_size             = ", rnn_inp_size)
    print("rnn_hid_size             = ", rnn_hid_size)
    print("num_rnn_layers           = ", num_rnn_layers)
    print("num_classes              = ", num_classes)
    print("th_x                     = ", th_x)
    print("th_h                     = ", th_h)
    print("Activation Integer  Bits = ", aqi)
    print("Activation Fraction Bits = ", aqf)
    print("Weight     Integer  Bits = ", wqi)
    print("Weight     Fraction Bits = ", wqf)
    num_rnn_layers = args.num_rnn_layers

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

    if mode != 0:
        print("Loading pretrained model: ", pretrain_model_path)
        pre_trained_model = torch.load(pretrain_model_path)
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

    # Print parameter count
    params = 0
    for param in list(net.parameters()):
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        params += sizes
    print('::: # network parameters: ' + str(params))

    # Select Optimizer
    optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=lr)

    # Select Loss Function
    criterion = nn.L1Loss()  # L1 loss function
    # criterion = nn.MSELoss()  # Mean square error loss function

    ########################################################
    # Training
    ########################################################
    if mode == 0:  # Initialize Parameters only in Mode 0
        for name, param in net.named_parameters():
            print(name)
            if 'rnn' in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param)
            if 'fc' in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                    # nn.init.xavier_uniform_(param)
            if 'bias' in name:  # all biases
                nn.init.constant_(param, 0)

    # Epoch loop
    print("Starting training...")

    # Training Log
    dict_history = {}
    dict_log = {}
    dict_history['epoch'] = []
    dict_history['time'] = []
    dict_history['lr'] = []
    dict_history['train_loss'] = []
    dict_history['dev_loss'] = []
    dict_history['dev_gain'] = []
    dict_history['test_loss'] = []
    dict_history['sp_W'] = []
    dict_history['sp_dx'] = []
    dict_history['sp_dh'] = []
    dev_gain = 1

    # Timer
    start_time = time.time()
    for epoch in range(num_epochs):

        ###########################################################################################################
        # Training - Iterate batches
        ###########################################################################################################
        net = net.train()
        train_loss = 0
        train_batches = 0
        sp_dx = 0
        sp_dh = 0
        net.set_quantize_act(1)  # Enable quantization in activation (only applies to DeltaGRU)
        net.set_eval_sparsity(0)  # Don't evaluate sparsity for faster training

        for batch, labels in tqdm(train_generator):  # Iterate through batches
            # Move data to GPU
            batch = batch.float().cuda().transpose(0, 1)
            labels = labels.float().cuda()
            batch = quantizeTensor(batch, aqi, aqf, qa)

            # Optimization
            optimizer.zero_grad()

            # Forward propagation
            # GRU Input size must be (seq_len, batch, input_size)
            out, _, _ = net(batch)

            # Get loss
            loss = criterion(out, labels)

            # Backward propagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 100)

            # Update parameters
            optimizer.step()

            # Increment monitoring variables
            batch_loss = loss.detach()
            train_loss += batch_loss  # Accumulate loss2
            train_batches += 1  # Accumulate count so we can calculate mean later

        # Quantize the RNN weights after every epoch
        net = quantize_rnn(net, wqi, wqf, 1)

        # Evaluate Weight Sparsity
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            # if 'rnn' in name:
            if 'weight' in name:
                n_nonzero_weight_elem += len(param.data.nonzero())
                n_weight_elem += param.data.nelement()
        sp_W = 1 - (n_nonzero_weight_elem / n_weight_elem)

        ###########################################################################################################
        # Validation - Iterate batches
        ###########################################################################################################
        net = net.eval()
        net.set_quantize_act(1)
        net.set_eval_sparsity(1)  # Enable sparsity evaluation

        dev_loss = 0
        dev_batches = 0

        for (batch, labels) in tqdm(dev_generator):
            # Move data to GPU
            batch = batch.float().cuda().transpose(0, 1)
            labels = labels.float().cuda()
            batch = quantizeTensor(batch, aqi, aqf, qa)

            # Forward propagation
            # GRU Input size must be (look_back_len, batch, input_size)
            out, _, _ = net(batch)

            # Get loss
            loss = criterion(out, labels)

            # Increment monitoring variables
            batch_loss = loss.detach()
            dev_loss += batch_loss
            dev_batches += 1  # Accumulate count so we can calculate mean later

        ###########################################################################################################
        # Test - Iterate batches
        ###########################################################################################################
        net = net.eval()
        net.set_quantize_act(1)
        net.set_eval_sparsity(1)  # Enable sparsity evaluation

        nz_dx = 0
        nz_dh = 0

        test_loss = 0
        test_batches = 0
        # Batch loop - validation
        for (batch, labels) in tqdm(test_generator):
            # Move data to GPU
            batch = batch.float().cuda().transpose(0, 1)
            labels = labels.float().cuda()
            batch = quantizeTensor(batch, aqi, aqf, qa)

            # Forward propagation
            if "Delta" in rnn_type:
                # GRU Input size must be (look_back_len, batch, input_size)
                out, _, _ = net(batch)
                # Get Statistics
                all_dx = net.all_layer_dx
                all_dh = net.all_layer_dh
                # Evaluate Temporal sparsity
                seq_len = t.ones(batch.size(1)).int() * (batch.size(0) - 1)
                sp_dx += get_temporal_sparsity(all_dx, seq_len, th_x)
                sp_dh += get_temporal_sparsity(all_dh, seq_len, th_h)
            else:
                out, _, _ = net(batch)
            if rnn_type != "FC":
                out = out[:, -1, :].squeeze()
                labels = labels[:, -1, :].squeeze()
            loss = criterion(out, labels)

            # Increment monitoring variables
            batch_loss = loss.detach()
            test_loss += batch_loss
            test_batches += 1  # Accumulate count so we can calculate mean later

        # Get Temporal Sparsity
        sp_dx = sp_dx / dev_batches
        sp_dh = sp_dh / dev_batches

        # Get current learning rate
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

        # Get History
        dict_history['epoch'].append(epoch)
        dict_history['time'].append(timeSince(start_time))
        dict_history['lr'].append(lr_curr)
        dict_history['train_loss'].append(train_loss.detach().cpu().numpy() / train_batches)
        dict_history['dev_loss'].append(dev_loss.detach().cpu().numpy() / dev_batches)
        dict_history['test_loss'].append(test_loss.detach().cpu().numpy() / test_batches)
        dict_history['sp_W'].append(sp_W)
        dict_history['sp_dx'].append(sp_dx)
        dict_history['sp_dh'].append(sp_dh)

        # Get relative dev loss gain
        if epoch >= 1:
            dev_gain = (dict_history['dev_loss'][epoch - 1] - dict_history['dev_loss'][epoch]) / \
                       dict_history['dev_loss'][epoch - 1]
        dict_history['dev_gain'].append(dev_gain)

        # Get Log for write
        dict_log['epoch'] = epoch
        dict_log['time'] = timeSince(start_time)
        dict_log['params'] = params
        dict_log['lr'] = lr_curr
        dict_log['train_loss'] = train_loss.detach().cpu().numpy() / train_batches
        dict_log['dev_loss'] = dev_loss.detach().cpu().numpy() / dev_batches
        dict_log['dev_gain'] = dev_gain
        dict_log['test_loss'] = test_loss.detach().cpu().numpy() / test_batches
        dict_log['sp_W'] = sp_W
        dict_log['sp_dx'] = sp_dx
        dict_log['sp_dh'] = sp_dh


        print('Epoch: %3d of %3d | '
              'Time: %s | '
              'LR: %1.5f | '
              'Train-L: %6.4f | '
              'Val-L: %6.4f | '
              'Val-Gain: %3.2f |'
              'Test-L: %6.4f | '
              'Sp.W: %2.2f | '
              'Sp.X: %2.2f | '
              'Sp.H: %2.2f | ' % (dict_history['epoch'][epoch], num_epochs - 1,
                                  dict_history['time'][epoch],
                                  dict_history['lr'][epoch],
                                  dict_history['train_loss'][epoch],
                                  dict_history['dev_loss'][epoch],
                                  dict_history['dev_gain'][epoch] * 100,
                                  dict_history['test_loss'][epoch],
                                  dict_history['sp_W'][epoch] * 100,
                                  dict_history['sp_dx'][epoch] * 100,
                                  dict_history['sp_dh'][epoch] * 100))

        # Save the best model with the lowest dev loss
        if epoch == 0:
            min_dev_loss = dev_loss
        if dev_loss <= min_dev_loss:
            min_dev_loss = dev_loss
            torch.save(net.state_dict(), savepath)
            print('>>> saving best model from epoch {}'.format(epoch))

        # Write Log File
        write_log(logpath, dict_log)

    print("Training Completed...                                               ")
    print(" ")
