import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy

class ConvBlock(nn.Module):
    def __init__(self, n_input_feature_maps, n_output_feature_maps, kernel_size_2d, batch_norm = False, pool_stride = None):
        super(ConvBlock, self).__init__()
        assert all(x % 2 == 1 for x in kernel_size_2d)
        self.n_input = n_input_feature_maps
        self.n_output = n_output_feature_maps
        self.kernel_size = kernel_size_2d
        self.batch_norm = batch_norm
        self.pool_stride = pool_stride
        # "~batch_norm" should be written as "not batch_norm"; otherwise ~True will evaluate to -2 and be treated as True.
        # But I'll keep this error to avoid breaking existing models.
        self.conv = nn.Conv2d(self.n_input, self.n_output, self.kernel_size, padding = tuple(x/2 for x in self.kernel_size), bias = ~batch_norm)
        if batch_norm: self.bn = nn.BatchNorm2d(self.n_output)
        nn.init.xavier_uniform(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm: x = self.bn(x)
        x = F.relu(x)
        if self.pool_stride is not None: x = F.max_pool2d(x, self.pool_stride)
        return x

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.__dict__.update(args.__dict__)     # Instill all args into self
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = 64
        self.output_size = 71 if self.mode == 'ctc' else 35
        self.conv = []
        pool_interval = self.n_conv_layers / self.n_pool_layers
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:        # this layer has pooling
                n_freq_bins /= 2
                n_output = self.embedding_size / n_freq_bins
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = self.embedding_size * 2 / n_freq_bins
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm = self.batch_norm, pool_stride = pool_stride)
            self.conv.append(layer)
            self.__setattr__('conv' + str(i + 1), layer)
            n_input = n_output
        self.gru = nn.GRU(self.embedding_size, self.embedding_size / 2, 1, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(self.embedding_size, self.output_size)
        # Better initialization
        nn.init.orthogonal(self.gru.weight_ih_l0); nn.init.constant(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0); nn.init.constant(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal(self.gru.weight_ih_l0_reverse); nn.init.constant(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0_reverse); nn.init.constant(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform(self.fc.weight); nn.init.constant(self.fc.bias, 0)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))                                               # x becomes (batch, channel, time, freq)
        for i in range(len(self.conv)):
            if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
            x = self.conv[i](x)                                                                 # x becomes (batch, channel, time, freq)
        x = x.permute(0, 2, 1, 3).contiguous()                                                  # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))                                      # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        x, _ = self.gru(x)                                                                      # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        if self.mode == 'ctc':
            log_prob = F.log_softmax(self.fc(x), dim = -1)                                      # shape of log_prob: (batch, time, output_size)
            return log_prob                                                                     # returns the log probability
        else:
            frame_prob = F.sigmoid(self.fc(x))                                                  # shape of frame_prob: (batch, time, output_size)
            frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
            return frame_prob

    def predict(self, x, batch_size = 300):
        # Predict in batches. Both input and output are numpy arrays.
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
                output = self.forward(input)
                result.append(output.data.cpu().numpy())
        return numpy.concatenate(result)
