# __author__     = "Chang Gao"
# __copyright__  = "Copyright 2018 to the author"
# __license__    = "Private"
# __version__    = "0.1.0"
# __maintainer__ = "Chang Gao"
# __email__      = "chang.gao@uzh.ch"
# __status__     = "Prototype"
import sys
import os
import torch as t
import torch.nn.functional as F
from torch.autograd.function import Function
import time
import math

def save_normalization(save_path, tr_mean, tr_std, lab_mean, lab_std):
    fn_base = os.path.splitext(save_path)[0]
    print("\nSaving normalization parameters to " + str(fn_base)+'-XX.pt')
    norm = {
            'tr_mean': tr_mean,
            'tr_std': tr_std,
            'lab_mean': lab_mean,
            'lab_std': lab_std,
        }
    t.save(norm, str(fn_base+'-norm.pt'))

def load_normalization(save_path):
    fn_base = os.path.splitext(save_path)[0]
    print("\nLoading normalization parameters from ", str(fn_base))
    norm = t.load(fn_base+'-norm.pt')
    return norm['tr_mean'], norm['tr_std'], norm['lab_mean'], norm['lab_std']

# print command line (maybe to use in a script)
def print_commandline(parser):
    args = parser.parse_args()
    print('Command line:')
    print('python '+os.path.basename(sys.argv[0]), end=' ')
    for arg in vars(args):
        print('--' + str(arg) + ' "' + str(getattr(args, arg))+'"', end=' ')
    print()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = t.typename(x).split('.')[-1]
    sparse_tensortype = getattr(t.sparse, x_typename)

    indices = t.nonzero(x)
    if indices.nelement() == 0:  # if all elements are zeros
        print("1", indices)
        return sparse_tensortype(*x.shape)
    else:
        print("2", indices)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())


def quantizeTensor(x, m, n, en):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """
    if en == 0:
        return x
    power = 2. ** n
    clip_val = 2. ** (m + n - 1)
    value = t.round(x * power)
    # value = GradPreserveRoundOp.apply(x * power)  # rounding
    value = t.clamp(value, -clip_val, clip_val - 1)  # saturation arithmetic
    value = value / power
    return value


def quantize_rnn(net, qi, qf, en):
    for name, param in net.named_parameters():
        if 'rnn' in name:
            param.data = quantizeTensor(param.data, qi, qf, en)
    return net


def pruneTensor(x, alpha):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """

    n_neuron = x.size(0)
    n_input = x.size(1)
    prune_prob_mask = t.exp(-alpha * t.unsqueeze(t.arange(0, n_neuron), dim=1).repeat(1, n_input).float()).cuda()
    prune_rand_mask = t.rand(n_neuron, n_input).cuda()
    prune_mask = prune_rand_mask.masked_fill_(prune_rand_mask > prune_prob_mask, 1)
    prune_mask = prune_mask.masked_fill_(prune_rand_mask <= prune_prob_mask, 0)
    _, indices = t.sort(t.abs(x), 0)
    # print("indices shape", indices.size())
    # print("prune_mask shape", prune_mask.size())
    # print("x shape", x.size())
    for j in range(0, n_input):
        x[indices[:, j], j] *= prune_mask[:, j]

    return x


def targetedDropout(x, gamma, alpha, epoch):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """

    t.manual_seed(epoch)
    t.cuda.manual_seed_all(epoch)

    n_elements = x.numel()
    drop_part = round(n_elements * gamma)
    weight_vec = x.view(-1)
    weight_vec_abs = t.abs(weight_vec)
    sorted, indices = t.sort(weight_vec_abs)
    # print(sorted)
    drop_indices = indices[0:drop_part]
    drop_rand_mask = t.rand(drop_indices.size(0)).cuda()
    drop_mask = t.ones(drop_indices.size(0)).cuda()
    drop_mask = drop_mask.masked_fill_(drop_rand_mask <= alpha, 0)
    weight_vec[drop_indices] *= drop_mask
    weight = t.reshape(weight_vec, (x.size(0), x.size(1)))

    return weight


def alignedTargetedDropout(x, gamma, alpha, num_pe, epoch):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """

    n_rows = x.shape[0]
    n_cols = x.shape[1]

    # Split and shuffle weight matrix
    for i in range(0, num_pe):
        for j in range(0, n_cols):
            targetedDropout(x[np.arange(i, n_rows, num_pe), j], gamma, alpha, epoch)
    return x


class GradPreserveRoundOp(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        output = t.round(input)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_input = grad_output

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # print(grad_output.size())
        # if not t.equal(grad_output, QuantizeT(grad_output, dW_qp)): print("grad_output not quantized")
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        # Return same number of parameters as "def forward(...)"
        return grad_input


class GradPreserveThreshold(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, threshold, value):
        output = F.threshold(input, threshold, value)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_input = grad_output

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # print(grad_output.size())
        # if not t.equal(grad_output, QuantizeT(grad_output, dW_qp)): print("grad_output not quantized")
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        # Return same number of parameters as "def forward(...)"
        return grad_input


def look_ahead_seq(seq_in, t_width=16, padding=0, batch_first=0):
    # Convert input sequence to batch first shape (seq_len, n_batch, n_feature)
    seq = seq_in
    if batch_first:
        seq = seq_in.transpose(0, 1)

    seq_len = seq.size(0)
    n_batch = seq.size(1)
    n_feature = seq.size(2)
    # int(t.ceil(float(seq_len)/float(t_width)))
    new_seq = []
    for i in range(0, seq_len):
        if i < seq_len - t_width:
            seq_block = seq[i:i + t_width, :, :]
        else:
            seq_block = seq[i:, :, :]
            seq_block_pad = t.zeros([t_width - (seq_len - i), n_batch, n_feature], dtype=t.float32).cuda()
            seq_block = t.cat((seq_block, seq_block_pad), 0)
        new_seq.append(seq_block)
    new_seq = t.stack(new_seq, 0)
    new_seq = new_seq.transpose(1, 2)
    new_seq = new_seq.transpose(0, 1)
    new_seq = new_seq.transpose(2, 3)
    return new_seq


def look_around_seq(seq_in, t_width=16, padding=0, batch_first=0):
    # Convert input sequence to batch first shape (seq_len, n_batch, n_feature)
    seq = seq_in
    if batch_first:
        seq = seq_in.transpose(0, 1)

    seq_len = seq.size(0)
    n_batch = seq.size(1)
    n_feature = seq.size(2)
    # int(t.ceil(float(seq_len)/float(t_width)))
    new_seq = []
    for i in range(0, seq_len):
        if i >= seq_len - t_width:
            seq_block = seq[i - t_width:, :, :]
            seq_block_pad = t.zeros([t_width - (seq_len - i) + 1, n_batch, n_feature], dtype=t.float32).cuda()
            seq_block = t.cat((seq_block, seq_block_pad), 0)
        elif i < t_width:
            seq_block = seq[0:i + 1 + t_width, :, :]
            seq_block_pad = t.zeros([t_width - i, n_batch, n_feature], dtype=t.float32).cuda()
            seq_block = t.cat((seq_block, seq_block_pad), 0)
        else:
            seq_block = seq[i - t_width:i + 1 + t_width, :, :]
        # print(seq_block.size())
        new_seq.append(seq_block)
    new_seq = t.stack(new_seq, 0)
    new_seq = new_seq.transpose(1, 2)
    new_seq = new_seq.transpose(0, 1)
    new_seq = new_seq.transpose(2, 3)
    return new_seq


def get_temporal_sparsity(list_layer, seq_len, threshold):
    # Evaluate Sparsity
    num_zeros = 0
    num_elems = 0
    # print(seq_len.size())
    # Iterate through layers
    for layer in list_layer:
        all_delta_vec = layer.transpose(0, 1)
        all_delta_vec = t.abs(all_delta_vec)  # Take absolute values of all delta vector elements
        for i, delta_vec in enumerate(all_delta_vec):
            seq = delta_vec[:seq_len[i], :]
            zero_mask = seq < threshold
            num_zeros += t.sum(zero_mask)
            num_elems += t.numel(zero_mask)
    sparsity = float(num_zeros) / float(num_elems)
    return sparsity
