import torch.nn as nn
from modules.deltarnn import DeltaGRU
import torch.nn.functional as F
# Definition of a neural network model

class Model(nn.Module):
    def __init__(self,
                 inp_size,
                 cla_type,
                 cla_size,
                 cla_layers,
                 num_classes,
                 aqi=8,
                 aqf=8,
                 wqi=1,
                 wqf=7,
                 th_x=0,
                 th_h=0,
                 eval_sparsity=0,
                 quantize_act=0,
                 debug=0,
                 cuda=1):

        super(Model, self).__init__()
        self.inp_size = inp_size
        self.cla_type = cla_type
        self.cla_layers = cla_layers
        self.cla_size = cla_size
        self.fc_size = 32
        self.aqi = aqi
        self.aqf = aqf
        self.wqi = wqi
        self.wqf = wqf
        self.th_x = th_x
        self.th_h = th_h
        self.p_dropout_rnn = 0
        self.eval_sparsity = eval_sparsity
        self.use_cuda = cuda
        self.quantize_act = quantize_act
        self.debug = debug

        # Statistics
        self.abs_delta_hid = 0
        self.abs_std_delta_hid = 0
        self.all_layer_dx = []
        self.all_layer_dh = []

        # Debug
        self.list_rnn_debug = []

        # self.bn = nn.BatchNorm1d(cla_size)

        # RNN
        self.rnn_type = cla_type
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.inp_size,
                              hidden_size=self.cla_size,
                              num_layers=self.cla_layers,
                              bias=True,
                              bidirectional=False,
                              dropout=0.5)
        elif self.rnn_type == 'DeltaGRU':
            self.rnn = DeltaGRU(n_inp=self.inp_size,
                                n_hid=self.cla_size,
                                num_layers=self.cla_layers,
                                th_x=self.th_x,
                                th_h=self.th_h,
                                aqi=self.aqi,
                                aqf=self.aqf,
                                wqi=self.wqi,
                                wqf=self.wqf,
                                eval_sparsity=self.eval_sparsity,
                                quantize_act=self.quantize_act,
                                debug=self.debug,
                                cuda=self.use_cuda)
        elif self.rnn_type == 'FC':
            self.rnn = nn.Sequential(
                nn.Linear(in_features=self.inp_size, out_features=self.cla_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=self.cla_size, out_features=self.cla_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.5),
             )
        self.fc = nn.Sequential(
            # nn.Linear(in_features=self.cla_size, out_features=self.cla_size, bias=True),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=self.cla_size, out_features=num_classes, bias=True)
        )

    def set_quantize_act(self, x):
        self.quantize_act = x
        if 'Delta' in self.cla_type:
            self.rnn.set_quantize_act(x)

    def set_eval_sparsity(self, x):
        self.eval_sparsity = x
        if 'Delta' in self.cla_type:
            self.rnn.set_eval_sparsity(x)

    def forward(self, x):
        self.list_rnn_debug = []

        if 'Delta' not in self.cla_type:
            if 'FC' not in self.cla_type:
                self.rnn.flatten_parameters()
        if self.cla_type == 'FC':
            x = x.transpose(0, 1)
            out_rnn = self.rnn(x)
            h_n = 0
        else:
            out_rnn, h_n = self.rnn(x)
            out_rnn = out_rnn.transpose(0, 1)
        # out_rnn = out_rnn.transpose(1, 2)
        # out_rnn = self.bn(out_rnn)
        # out_rnn = out_rnn.transpose(1, 2)
        out_fc = self.fc(out_rnn)

        if self.debug:
            self.list_rnn_debug = self.rnn.list_rnn_debug

        # Get Statistics
        if 'Delta' in self.cla_type:
            self.abs_delta_hid = self.rnn.all_layer_abs_delta_hid
            self.abs_delta_hid = self.abs_delta_hid.cpu()
            self.all_layer_dx = self.rnn.all_layer_dx
            self.all_layer_dh = self.rnn.all_layer_dh

        return out_fc, h_n, out_rnn

