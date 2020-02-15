import os
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
from modules.util import quantizeTensor, timeSince, quantize_rnn
from modules.log import write_log, write_log_header
from modules.ampro_data import load_ampro_data, Dataset
from modules.deltarnn import get_temporal_sparsity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GRU network.')
    parser.add_argument('--seed', default=2, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--prev_look_back_len', default=5, type=int, help='Max timesteps per batch.')
    parser.add_argument('--look_back_len', default=100, type=int, help='Max timesteps per batch.')
    parser.add_argument('--pred_len', default=0, type=int, help='Static batch size.')
    parser.add_argument('--batch_size', default=32, type=int, help='Static batch size.')
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs to train for.')
    parser.add_argument('--mode', default=0, type=int, help='Number of epochs to train for.')
    parser.add_argument('--num_rnn_layers', default=2, type=int, help='number of classification layers')
    parser.add_argument('--rnn_hid_size', default=128, type=int, help='Size of classification layer')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')  # 5e-4
    parser.add_argument('--qa', default=0, type=int, help='Whether quantize the network activations')
    parser.add_argument('--qw', default=1, type=int, help='Whether quantize the network weights')
    parser.add_argument('--aqi', default=8, type=int, help='Number of integer bits before decimal point')
    parser.add_argument('--aqf', default=8, type=int, help='Number of integer bits before decimal point')
    parser.add_argument('--wqi', default=1, type=int, help='Number of integer bits before decimal point')
    parser.add_argument('--wqf', default=7, type=int, help='Number of integer bits before decimal point')
    parser.add_argument('--th_x', default=4/256, type=float, help='Delta threshold for inputs')
    parser.add_argument('--th_h', default=128/256, type=float, help='Delta threshold for hidden states')
    parser.add_argument('--show_sp', default=0, type=int, help='Number of fraction bits after decimal point')
    parser.add_argument('--normalize', default=1, type=int, help='Best model used in testing, either "per", or "vloss"')
    args = parser.parse_args()

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
    prev_look_back_len = args.prev_look_back_len  # Length of history in timesteps used to train the network
    look_back_len = args.look_back_len            # Length of history in timesteps used to train the network
    pred_len = args.pred_len                      # Length of future in timesteps to predict
    lr = args.lr                                  # Learning rate
    batch_size = args.batch_size                  # Mini-batch size
    num_epochs = args.num_epochs                 # Number of epoches to train the network
    mode = args.mode

    print('###################################################################################\n\r'
          '# Hyperparameters\n\r'
          '###################################################################################')
    print("mode =               ", mode)
    print("prev_look_back_len = ", prev_look_back_len)
    print("look_back_len =      ", look_back_len)
    print("pred_len =           ", pred_len)
    print("lr =                 ", lr)
    print("batch_size =         ", batch_size)
    print("num_epochs =         ", num_epochs)

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
    if mode == 0:  # Pretrain on GRU
        str_target_variable = 'ankle_knee'
        str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size) + 'H-'
        filename = str_net_arch + str(rnn_type) + '-' + str(look_back_len) + 'T-' + str_target_variable
        pretrain_model_path = str_net_arch + 'GRU'
        logpath = './log/' + filename + '.csv'
        savepath = './save/' + filename + '.pt'
    elif mode == 1:  # Retrain on GRU
        str_target_variable = 'ankle_knee'
        str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size) + 'H-'

        filename = str_net_arch + str(rnn_type) + '-' + str(look_back_len) + 'T-' + str_target_variable
        pretrain_model_path = './save/' + str_net_arch + 'GRU' + '-' + str(prev_look_back_len) + 'T-' + str_target_variable + '.pt'
        logpath = './log/' + filename + '.csv'
        savepath = './save/' + filename + '.pt'
    elif mode == 2:  # Retrain on DeltaGRU
        str_target_variable = 'ankle_knee'
        str_net_arch = str(num_rnn_layers) + 'L-' + str(rnn_hid_size) + 'H-'
        filename = str_net_arch + str(rnn_type) + '-' + str(look_back_len) + 'T-' + str_target_variable
        pretrain_model_path = './save/' + str_net_arch + 'GRU' + '-' + str(look_back_len) + 'T-' + str_target_variable + '.pt'
        logpath = './log/' + filename + '_' + str(th_x) + '.csv'
        savepath = './save/' + filename + '_' + str(th_x) + '.pt'

    print("Loading pretrained model: ", pretrain_model_path)

    ########################################################
    # Create Dataset
    ########################################################
    _, ampro_data_1, labels_1 = load_ampro_data('./data/rachel_pd1.csv', look_back_len, pred_len)
    _, ampro_data_2, labels_2 = load_ampro_data('./data/rachel_pd2.csv', look_back_len, pred_len)
    _, ampro_data_3, labels_3 = load_ampro_data('./data/rachel_pd3.csv', look_back_len, pred_len)
    ampro_dev_sample, ampro_data_4, labels_4 = load_ampro_data('./data/rachel_pd4.csv', look_back_len, pred_len)
    _, ampro_data_5, labels_5 = load_ampro_data('./data/rachel_pd5.csv', look_back_len, pred_len)

    train_ampro_data = np.concatenate((ampro_data_1, ampro_data_2, ampro_data_3), axis=0)
    train_ampro_labels = np.concatenate((labels_1, labels_2, labels_3), axis=0)
    dev_ampro_data = ampro_data_4
    dev_ampro_labels = labels_4
    test_ampro_data = ampro_data_5
    test_ampro_labels = labels_5

    # Convert data to PyTorch Tensors
    train_data = torch.Tensor(train_ampro_data).float()
    train_labels = torch.Tensor(train_ampro_labels).float()
    dev_data = torch.Tensor(dev_ampro_data).float()
    dev_labels = torch.Tensor(dev_ampro_labels).float()
    test_data = torch.Tensor(test_ampro_data).float()
    test_labels = torch.Tensor(test_ampro_labels).float()

    # Normalize data
    mean_train_data = torch.mean(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)
    std_train_data = torch.std(train_data.reshape(train_data.size(0) * train_data.size(1), -1), 0)
    train_data_norm = (train_data - mean_train_data) / std_train_data
    dev_data_norm = (dev_data - mean_train_data) / std_train_data
    test_data_norm = (test_data - mean_train_data) / std_train_data

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

    if mode == 3:
        feat_size = train_data_norm.size(2)
        train_data_norm = t.reshape(train_data_norm, (-1, look_back_len*feat_size))
        dev_data_norm = t.reshape(dev_data_norm, (-1, look_back_len*feat_size))
        test_data_norm = t.reshape(test_data_norm, (-1, look_back_len*feat_size))
        train_labels = train_labels[:, -1:, :].squeeze()
        dev_labels = dev_labels[:, -1:, :].squeeze()
        test_labels = test_labels[:, -1:, :].squeeze()

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
    criterion = nn.L1Loss()  # Mean square error loss function
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
            # GRU Input size must be (look_back_len, batch, input_size)
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
