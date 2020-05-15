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
from modules.data import load_data, Dataset, normalize, unnormalize
from modules.deltarnn import get_temporal_sparsity
from modules import parseArgs
import warnings


from memory_profiler import profile
import timeit

#@profile(precision=4)
def train_network():

    start = timeit.default_timer()

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
    cw_plen = args.cw_plen  # Length of history in timesteps used to train the network
    cw_flen = args.cw_flen  # Length of future in timesteps to predict
    pw_len = args.pw_len  # Offset of future in timesteps to predict
    pw_off = args.pw_off  # Length of future in timesteps to predict
    seq_len = args.seq_len  # Sequence length
    if seq_len<=cw_plen: warnings.warn('sequence length '+str(seq_len)+' is less than context window length '+str(cw_plen))
    lr = args.lr  # Learning rate
    batch_size = args.batch_size  # Mini-batch size
    num_epochs = args.num_epochs  # Number of epoches to train the network
    mode = args.mode
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
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
    save_path = {}
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
        pretrain_model_path = './save/' + str_net_arch + 'GRU' + '-' + str_windows + str_target_variable + '.pt'  # TODO probably a bug in this filename, has extra parts
        logpath = './log/' + filename + '_' + str(th_x) + '.csv'
        savepath = './save/' + filename + '_' + str(th_x) + '.pt'

    ########################################################
    # Create Dataset
    ########################################################
    train_features, train_dict, train_targets, target_dict,_,actual_dict, mean_train_features, std_train_features, mean_train_targets, std_train_targets = \
        load_data(train_file, cw_plen, cw_flen, pw_len, pw_off, seq_len, args.stride, args.med_filt, args.cutoff_hz)
    dev_features,_, dev_targets, _, _,_, _, _, _, _ = \
        load_data(val_file, cw_plen, cw_flen, pw_len, pw_off, seq_len, args.stride, args.med_filt, args.cutoff_hz)
    test_features,_, test_targets,_,_, _, _, _, _, _ = \
        load_data(test_file, cw_plen, cw_flen, pw_len, pw_off, seq_len, args.stride, args.med_filt, args.cutoff_hz)

    save_normalization(savepath, mean_train_features, std_train_features, mean_train_targets, std_train_targets)

    # normalize all data by training set values
    train_features = normalize(train_features, mean_train_features, std_train_features)
    train_targets = normalize(train_targets, mean_train_targets, std_train_targets)
    dev_features = normalize(dev_features, mean_train_features, std_train_features)
    dev_targets = normalize(dev_targets, mean_train_targets, std_train_targets)
    test_features = normalize(test_features, mean_train_features, std_train_features)
    test_targets = normalize(test_targets, mean_train_targets, std_train_targets)
    

    print("train_features: ", train_features.shape)
    print("mean_train_features: ", mean_train_features)
    print("std_train_features: ", std_train_features)

    # Convert Numpy Arrays to PyTorch Tensors
    # train_features = torch.from_numpy(train_features).float()
    # train_targets = torch.from_numpy(train_targets).float()
    # dev_features = torch.from_numpy(dev_features).float()
    # dev_targets = torch.from_numpy(dev_targets).float()
    # test_features = torch.from_numpy(test_features).float()
    # test_targets = torch.from_numpy(test_targets).float()

    # Create PyTorch Dataset
    train_set = Dataset(train_features, train_targets, args)
    dev_set = Dataset(dev_features, dev_targets, args)
    test_set = Dataset(test_features, test_targets, args)

    # Create PyTorch dataloaders for train and dev set
    train_generator = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers = args.num_workers)
    dev_generator = data.DataLoader(dataset=dev_set, batch_size=512, shuffle=False, num_workers = args.num_workers)
    test_generator = data.DataLoader(dataset=test_set, batch_size=512, shuffle=False, num_workers = args.num_workers)

    # Get number of classes
    X,y  = train_set.__getitem__(1)
    num_classes = y.size(-1)
    print("\n")

    # print('###################################################################################\n\r'
    #       '# Dataset (Normalized)\n\r'
    #       '###################################################################################')
    # print("# Train Data  | Size:   %s                            \n"
    #       "#             | Min:    %f                            \n"
    #       "#             | Mean:   %f                            \n"
    #       "#             | Median: %f                            \n"
    #       "#             | Max:    %f                            \n"
    #       "#             | Std:    %f                            \n"
    #       "#-----------------------------------------------------\n"
    #       "# Dev   Data  | Size:   %s                            \n"
    #       "#             | Min:    %f                            \n"
    #       "#             | Mean:   %f                            \n"
    #       "#             | Median: %f                            \n"
    #       "#             | Max:    %f                            \n"
    #       "#             | Std:    %f                            \n"
    #       "#-----------------------------------------------------\n"
    #       "# Test  Data  | Size:   %s                            \n"
    #       "#             | Min:    %f                            \n"
    #       "#             | Mean:   %f                            \n"
    #       "#             | Median: %f                            \n"
    #       "#             | Max:    %f                            \n"
    #       "#             | Std:    %f                            \n"
    #       "#-----------------------------------------------------\n"
    #       "# Train Label | Size:   %s                            \n"
    #       "#             | Min:    %f                            \n"
    #       "#             | Mean:   %f                            \n"
    #       "#             | Median: %f                            \n"
    #       "#             | Max:    %f                            \n"
    #       "#             | Std:    %f                            \n"
    #       "#-----------------------------------------------------\n"
    #       "# Dev   Label | Size:   %s                            \n"
    #       "#             | Min:    %f                            \n"
    #       "#             | Mean:   %f                            \n"
    #       "#             | Median: %f                            \n"
    #       "#             | Max:    %f                            \n"
    #       "#             | Std:    %f                            \n"
    #       "#-----------------------------------------------------\n"
    #       "# Test  Label | Size:   %s                            \n"
    #       "#             | Min:    %f                            \n"
    #       "#             | Mean:   %f                            \n"
    #       "#             | Median: %f                            \n"
    #       "#             | Max:    %f                            \n"
    #       "#             | Std:    %f                            \n"
    #       "#-----------------------------------------------------\n"
    #       % (str(list(train_features.size())), train_features.min(), train_features.mean(), train_features.median(), train_features.max(), train_features.std(),
    #          str(list(dev_features.size())), dev_features.min(), dev_features.mean(), dev_features.median(), dev_features.max(), dev_features.std(),
    #          str(list(test_features.size())), test_features.min(), test_features.mean(), test_features.median(), test_features.max(), test_features.std(),
    #          str(list(train_targets.size())), train_targets.min(), train_targets.mean(), train_targets.median(), train_targets.max(), train_targets.std(),
    #          str(list(dev_targets.size())), dev_targets.min(), dev_targets.mean(), dev_targets.median(), dev_targets.max(), dev_targets.std(),
    #          str(list(test_targets.size())), test_targets.min(), test_targets.mean(), test_targets.median(), test_targets.max(), test_targets.std()
    #         )
    #       )

    print("\n\r")

    # # Data Range
    # angle_min = 0
    # angle_max = 2 * math.pi
    # position_min = -655
    # position_max = 1644

    print('###################################################################################\n\r'
          '# Network\n\r'
          '###################################################################################')
    # Network Dimension
    rnn_inp_size = X.size(-1)
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
                       cuda=args.cuda)

    if mode != 0:
        print("Loading pretrained model: ", pretrain_model_path)
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
    if args.cuda:
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
    # criterion = nn.L1Loss()  # L1 loss function
    criterion = nn.MSELoss()  # Mean square error loss function

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
        net.set_quantize_act(0)  # Enable quantization in activation (only applies to DeltaGRU)
        net.set_eval_sparsity(0)  # Don't evaluate sparsity for faster training

        for batch, labels in tqdm(train_generator):  # Iterate through batches
            # Move data to GPU
            
            np_batch = batch.numpy()
            np_label = labels.numpy()
            
            if args.cuda:
                batch = batch.float().cuda().transpose(0, 1)
                labels = labels.float().cuda()
            else:
                batch = batch.float().transpose(0, 1)
                labels = labels.float()

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
        net = quantize_rnn(net, wqi, wqf, qw)

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
        net.set_quantize_act(qa)
        net.set_eval_sparsity(1)  # Enable sparsity evaluation

        dev_loss = 0
        dev_batches = 0

        for (batch, labels) in tqdm(dev_generator):
            if args.cuda:
                batch = batch.float().cuda().transpose(0, 1)
                labels = labels.float().cuda()
            else:
                batch = batch.float().transpose(0, 1)
                labels = labels.float()
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
        net.set_quantize_act(qa)
        net.set_eval_sparsity(1)  # Enable sparsity evaluation

        nz_dx = 0
        nz_dh = 0

        test_loss = 0
        test_batches = 0
        # Batch loop - validation
        for (batch, labels) in tqdm(test_generator):
            if args.cuda:
                batch = batch.float().cuda().transpose(0, 1)
                labels = labels.float().cuda()
            else:
                batch = batch.float().transpose(0, 1)
                labels = labels.float()
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

        print('\nEpoch: %3d of %3d | '
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

    stop = timeit.default_timer() 
    total_time = stop-start
    return total_time

if __name__ == '__main__':
  total_time = train_network()
  print('Total time of training the network: '+str(total_time))