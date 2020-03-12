import os
import sys
import collections
import argparse
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
from modules.util import quantizeTensor, timeSince, quantize_rnn, print_commandline,save_normalization
from modules.log import write_log, write_log_header
from modules.data import load_data, Dataset
from modules.deltarnn import get_temporal_sparsity

TRAIN_FILE_DEFAULT='data/cartpole-2020-03-09-14-43-54 stock motor PD control w dance and steps.csv'
VAL_FILE_DEFAULT='data/cartpole-2020-03-09-14-21-24 stock motor PD angle zero correct.csv'
TEST_FILE_DEFAULT='data/cartpole-2020-03-09-14-24-21 stock motor PD with dance.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    parser.add_argument('--train_file', default=TRAIN_FILE_DEFAULT, type=str,help='Training dataset file')
    parser.add_argument('--val_file', default=VAL_FILE_DEFAULT, type=str,help='Validation dataset file')
    parser.add_argument('--test_file', default=TEST_FILE_DEFAULT, type=str,help='Testing dataset file')
    parser.add_argument('--seed', default=1, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--cw_plen', default=5, type=int, help='Number of previous timesteps in the context window, leads to initial latency')
    parser.add_argument('--cw_flen', default=0, type=int, help='Number of future timesteps in the context window, leads to consistent latency')
    parser.add_argument('--pw_len', default=20, type=int, help='Number of future timesteps in the prediction window')
    parser.add_argument('--pw_off', default=1, type=int, help='Offset in #timesteps of the prediction window w.r.t the current timestep')
    parser.add_argument('--seq_len', default=100, type=int, help='Sequence Length for BPTT training; samples are drawn with this length randomly throughout training set')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size. How many samples to run forward in parallel before each weight update.')
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs to train for.')
    parser.add_argument('--mode', default=1, type=int, help='Mode 0 - Pretrain on GRU; Mode 1 - Retrain on GRU; Mode 2 - Retrain on DeltaGRU')
    parser.add_argument('--num_rnn_layers', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn_hid_size', default=128, type=int, help='RNN Hidden layer size')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')  # 5e-4
    parser.add_argument('--qa', default=0, type=int, help='Whether quantize the network activations')
    parser.add_argument('--qw', default=0, type=int, help='Whether quantize the network weights')
    parser.add_argument('--aqi', default=8, type=int, help='Number of integer bits before decimal point for activation')
    parser.add_argument('--aqf', default=8, type=int, help='Number of integer bits after decimal point for activation')
    parser.add_argument('--wqi', default=8, type=int, help='Number of integer bits before decimal point for weight')
    parser.add_argument('--wqf', default=8, type=int, help='Number of integer bits after decimal point for weight')
    parser.add_argument('--th_x', default=64/256, type=float, help='Delta threshold for inputs')
    parser.add_argument('--th_h', default=64/256, type=float, help='Delta threshold for hidden states')
    args = parser.parse_args()

    print_commandline(parser)

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
    _, data_1, labels_1 = load_data(train_file,
                                    cw_plen, cw_flen, pw_len, pw_off, seq_len)
    _, data_2, labels_2 = load_data(val_file, cw_plen,
                                    cw_flen, pw_len, pw_off, seq_len)
    _, data_3, labels_3 = load_data(test_file, cw_plen, cw_flen,
                                    pw_len, pw_off, seq_len)

    train_ampro_data = data_1 #np.concatenate((data_1), axis=0) # if only one file, then don't concatenate, it kills an axis
    train_ampro_labels = labels_1 #np.concatenate((labels_1), axis=0)
    dev_ampro_data = data_2
    dev_ampro_labels = labels_2
    test_ampro_data = data_3
    test_ampro_labels = labels_3

    # Convert data to PyTorch Tensors
    train_data = torch.Tensor(train_ampro_data).float()
    train_labels = torch.Tensor(train_ampro_labels).float()
    dev_data = torch.Tensor(dev_ampro_data).float()
    dev_labels = torch.Tensor(dev_ampro_labels).float()
    test_data = torch.Tensor(test_ampro_data).float()
    test_labels = torch.Tensor(test_ampro_labels).float()

    # Normalize data TODO save these parameters to use for inference
    mean_train_data = torch.mean(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)
    std_train_data = torch.std(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)
    train_data_norm = (train_data - mean_train_data) / std_train_data
    dev_data_norm = (dev_data - mean_train_data) / std_train_data
    test_data_norm = (test_data - mean_train_data) / std_train_data
    save_normalization(savepath,mean_train_data,std_train_data)

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
    train_set = Dataset(train_data_norm, train_labels, mode)
    dev_set = Dataset(dev_data_norm, dev_labels, mode)
    test_set = Dataset(test_data_norm, test_labels, mode)

    # Create PyTorch dataloaders for train and dev set
    train_generator = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    dev_generator = data.DataLoader(dataset=dev_set, batch_size=512, shuffle=True)
    test_generator = data.DataLoader(dataset=test_set, batch_size=512, shuffle=True)

    print('###################################################################################\n\r'
          '# Dataset\n\r'
          '###################################################################################')
    print("Dim: (num_sample, look_back_len, feat_size)")
    print("Train data size:  ", train_data_norm.size())
    print("Train label size: ", train_labels.size())
    print("Test data size:   ", dev_data_norm.size())
    print("Test label size:  ", dev_labels.size())
    print("\n\r")

    print('###################################################################################\n\r'
          '# Network\n\r'
          '###################################################################################')
    # Network Dimension
    rnn_inp_size = train_data_norm.size(-1)
    print("rnn_inp_size = ", rnn_inp_size)
    print("rnn_hid_size = ", rnn_hid_size)
    print("num_rnn_layers = ", num_rnn_layers)
    print("num_classes  = ", num_classes)
    print("th_x  = ", th_x)
    print("th_h  = ", th_h)
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
            batch = quantizeTensor(batch, aqi, aqf, 1)

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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 200)

            # Update parameters
            optimizer.step()

            # Increment monitoring variables
            batch_loss = loss.detach()
            train_loss += batch_loss  # Accumulate loss
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
            batch = quantizeTensor(batch, aqi, aqf, 1)

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
            batch = quantizeTensor(batch, aqi, aqf, 1)

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
              'LR: %1.8f | '
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
