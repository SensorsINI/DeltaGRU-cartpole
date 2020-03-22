import argparse
import main
from modules.util import  print_commandline
import warnings

TRAIN_FILE_DEFAULT= '../data/cartpole-2020-03-09-14-43-54 stock motor PD control w dance and steps.csv'
VAL_FILE_DEFAULT= '../data/cartpole-2020-03-09-14-21-24 stock motor PD angle zero correct.csv'
TEST_FILE_DEFAULT= '../data/cartpole-2020-03-09-14-24-21 stock motor PD with dance.csv'

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

    # which hardware to usee
    parser.add_argument('--cuda', default=0, type=int, help='1 to use cuda, 0 for CPU (better debug output)')  # 5e-4
    # data
    parser.add_argument('--train_file', default=TRAIN_FILE_DEFAULT, type=str, help='Training dataset file')
    parser.add_argument('--val_file',   default=VAL_FILE_DEFAULT, type=str, help='Validation dataset file')
    parser.add_argument('--test_file',  default=TEST_FILE_DEFAULT, type=str, help='Testing dataset file')
    #training
    parser.add_argument('--mode',       default=1, type=int,  help='Mode 0 - Pretrain on GRU; Mode 1 - Retrain on GRU; Mode 2 - Retrain on DeltaGRU')
    parser.add_argument('--seed',       default=1, type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--stride',     default=1, type=int, help='Stride for time series data slice window')
    parser.add_argument('--seq_len',    default=32, type=int,  help='Sequence Length for BPTT training; samples are drawn with this length randomly throughout training set')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size. How many samples to run forward in parallel before each weight update.')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs to train for.')
    parser.add_argument('--lr',                 default=1e-4, type=float, help='Learning rate')  # 5e-4
    #architecure
    parser.add_argument('--rnn_type',   default='GRU',  help='Mode 0 - Pretrain on GRU; Mode 1 - Retrain on GRU; Mode 2 - Retrain on DeltaGRU')
    parser.add_argument('--num_rnn_layers',     default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn_hid_size',       default=32, type=int, help='RNN Hidden layer size')
    parser.add_argument('--cw_plen',    default=10, type=int, help='Number of previous timesteps in the context window, leads to initial latency')
    parser.add_argument('--cw_flen',    default=0, type=int,  help='Number of future timesteps in the context window, leads to consistent latency')
    parser.add_argument('--pw_len',     default=10, type=int, help='Number of future timesteps in the prediction window')
    parser.add_argument('--pw_off',     default=1, type=int,  help='Offset in #timesteps of the prediction window w.r.t the current timestep')
    #plotting
    parser.add_argument('--pw_idx',     default=1, type=int, help='Index of timestep in the prediction window to show in plots')
    parser.add_argument('--plot_len',   default=500, type=int, help='Number of timesteps in the plot window')
    # quantization
    parser.add_argument('--qa',     default=0, type=int, help='Whether quantize the network activations')
    parser.add_argument('--qw',     default=0, type=int, help='Whether quantize the network weights')
    parser.add_argument('--aqi',    default=8, type=int, help='Number of integer bits before decimal point for activation')
    parser.add_argument('--aqf',    default=8, type=int, help='Number of integer bits after decimal point for activation')
    parser.add_argument('--wqi',    default=8, type=int, help='Number of integer bits before decimal point for weight')
    parser.add_argument('--wqf',    default=8, type=int, help='Number of integer bits after decimal point for weight')
    # delta RNN
    parser.add_argument('--th_x',   default=64 / 256, type=float, help='Delta threshold for inputs')
    parser.add_argument('--th_h',   default=64 / 256, type=float, help='Delta threshold for hidden states')

    args = parser.parse_args()
    print_commandline(parser)
    if not args.cuda:warnings.warn('not using CUDA hardware - see --cuda 0|1')
    return args
