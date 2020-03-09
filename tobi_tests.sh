#!/bin/bash
# plot results
python test_plot.py --seed 1 --cw_plen 10 --cw_flen 0 --pw_len 100 --pw_off 1 --seq_len 100 --num_rnn_layers 2 --rnn_hid_size 128 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25
# at INI with stock motor files 9.3.20
main.py --seed 1 --cw_plen 1 --cw_flen 0 --pw_len 100 --pw_off 1 --seq_len 100 --batch_size 64 --num_epochs 5 --mode 0 --num_rnn_layers 2 --rnn_hid_size 128 --lr 0.0005 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25
python test_plot.py --seed 1 --cw_plen 1 --cw_flen 0 --pw_len 100 --pw_off 1 --seq_len 100  --num_rnn_layers 2 --rnn_hid_size 128 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25