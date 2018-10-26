import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.pooling = args.pooling
        self.dropout = args.dropout
        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding = (2, 2))                         # (1, 400, 64) -> (32, 400, 64)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), padding = (2, 2))                        # (32, 400, 32) -> (64, 400, 32)
        self.conv3 = nn.Conv2d(64, 128, (5, 5), padding = (2, 2))                       # (64, 200, 16) -> (128, 200, 16)
        self.gru = nn.GRU(1024, 100, 1, batch_first = True, bidirectional = True)
        self.fc_prob = nn.Linear(200, 17)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(200, 17)
        # Better initialization
        nn.init.xavier_uniform(self.conv1.weight); nn.init.constant(self.conv1.bias, 0)
        nn.init.xavier_uniform(self.conv2.weight); nn.init.constant(self.conv2.bias, 0)
        nn.init.xavier_uniform(self.conv3.weight); nn.init.constant(self.conv3.bias, 0)
        nn.init.orthogonal(self.gru.weight_ih_l0); nn.init.constant(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0); nn.init.constant(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal(self.gru.weight_ih_l0_reverse); nn.init.constant(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal(self.gru.weight_hh_l0_reverse); nn.init.constant(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform(self.fc_prob.weight); nn.init.constant(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform(self.fc_att.weight); nn.init.constant(self.fc_att.bias, 0)

    def forward(self, x):
        # shape of x: (batch, time, frequency) = (batch, 400, 64)
        x = x.view((-1, 1, x.size(1), x.size(2)))               # x becomes (batch, channel, time, frequency) = (batch, 1, 400, 64)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.max_pool2d(F.relu(self.conv1(x)), (1, 2))         # (batch, 32, 400, 32)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))         # (batch, 64, 200, 16)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))         # (batch, 128, 100, 8)
        x = x.permute(0, 2, 1, 3).contiguous()                  # x becomes (batch, time, channel, frequency) = (batch, 100, 128, 8)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))      # x becomes (batch, time, channel * frequency) = (batch, 100, 1024)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        x, _ = self.gru(x)                                          # (batch, 100, 200)
        if self.dropout > 0: x = F.dropout(x, p = self.dropout, training = self.training)
        frame_prob = F.sigmoid(self.fc_prob(x))                     # shape of frame_prob: (batch, time, class) = (batch, 100, 17)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, frame_prob, frame_att

    def predict(self, x, verbose = True, batch_size = 100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]
