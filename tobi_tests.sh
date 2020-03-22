#!/bin/bash
# plot results
python test_plot.py --seed 1 --cw_plen 10 --cw_flen 0 --pw_len 100 --pw_off 1 --seq_len 100 --num_rnn_layers 2 --rnn_hid_size 128 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25

## at INI with stock motor files 9.3.20
# 1 input sample, 100 sample prediction
# train
python main.py --seed 1 --cw_plen 1 --cw_flen 0 --pw_len 100 --pw_off 1 --seq_len 100 --batch_size 64 --num_epochs 5 --mode 0 --num_rnn_layers 2 --rnn_hid_size 128 --lr 0.0005 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25
# refine
python main.py --seed 1 --cw_plen 1 --cw_flen 0 --pw_len 100 --pw_off 1 --seq_len 100 --batch_size 64 --num_epochs 5 --mode 1 --num_rnn_layers 2 --rnn_hid_size 128 --lr 0.0005 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25

# plot
python test_plot.py --seed 1 --cw_plen 1 --cw_flen 0 --pw_len 100 --pw_off 1 --seq_len 100  --num_rnn_layers 2 --rnn_hid_size 128 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25

# 1 input sample, 20 sample 100ms prediction
# train
python main.py --seed 1 --cw_plen 5 --cw_flen 0 --pw_len 20 --pw_off 1 --seq_len 50 --batch_size 64 --num_epochs 10 --mode 0 --num_rnn_layers 2 --rnn_hid_size 128 --lr 0.0005 --qa 0 --qw 0 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25
# refine
python main.py --seed 1 --cw_plen 5 --cw_flen 0 --pw_len 20 --pw_off 1 --seq_len 50 --batch_size 64 --num_epochs 5 --mode 1 --num_rnn_layers 2 --rnn_hid_size 128 --lr 0.0005 --qa 0 --qw 0 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25

# plot
python test_plot.py --seed 1 --cw_plen 5 --cw_flen 0 --pw_len 20 --pw_off 1 --seq_len 50  --num_rnn_layers 2 --rnn_hid_size 128 --qa 0 --qw 1 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25

# shorter prediction
python main.py --seed 1 --cw_plen 5 --cw_flen 0 --pw_len 20 --pw_off 1 --seq_len 100 --batch_size 64 --num_epochs 5 --mode 1 --num_rnn_layers 2 --rnn_hid_size 128 --lr 0.0005 --qa 0 --qw 0 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25
python test_plot.py --seed 1 --cw_plen 5 --cw_flen 0 --pw_len 20 --pw_off 1 --seq_len 100  --num_rnn_layers 2 --rnn_hid_size 128 --qa 0 --qw 0 --aqi 8 --aqf 8 --wqi 8 --wqf 8 --th_x 0.25 --th_h 0.25


python main.py --train_file "data/cartpole-2020-03-09-14-43-54 stock motor PD control w dance and steps.csv" --val_file "data/cartpole-2020-03-09-14-21-24 stock motor PD angle zero correct.csv" --test_file "data/cartpole-2020-03-09-14-24-21 stock motor PD with dance.csv" --seed "1" --stride "1" --cw_plen "10" --cw_flen "0" --pw_len "10" --pw_off "1" --seq_len "32" --batch_size "32" --num_epochs "10" --mode "1" --num_rnn_layers "2" --rnn_hid_size "32" --lr "0.0001" --qa "0" --qw "0" --aqi "8" --aqf "8" --wqi "8" --wqf "8" --th_x "0.25" --th_h "0.25" --cuda "1"
python test_plot.py --train_file "data/cartpole-2020-03-09-14-43-54 stock motor PD control w dance and steps.csv" --val_file "data/cartpole-2020-03-09-14-21-24 stock motor PD angle zero correct.csv" --test_file "data/cartpole-2020-03-09-14-24-21 stock motor PD with dance.csv" --seed "1" --stride "1" --cw_plen "10" --cw_flen "0" --pw_len "10" --pw_off "1" --seq_len "32" --batch_size "32" --num_epochs "10" --mode "1" --num_rnn_layers "2" --rnn_hid_size "32" --lr "0.0001" --qa "0" --qw "0" --aqi "8" --aqf "8" --wqi "8" --wqf "8" --th_x "0.25" --th_h "0.25" --cuda "0"
