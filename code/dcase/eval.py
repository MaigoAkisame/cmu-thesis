import sys, os, os.path
import argparse
import numpy
from util_out import *
from util_f1 import *
from scipy.io import loadmat, savemat

# Parse input arguments
parser = argparse.ArgumentParser(description = '')
parser.add_argument('--pooling', type = str, default = 'lin', choices = ['max', 'ave', 'lin', 'exp', 'att'])
parser.add_argument('--dropout', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--ckpt_size', type = int, default = 500)
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd'])
parser.add_argument('--init_lr', type = float, default = 3e-4)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 0.5)
parser.add_argument('--random_seed', type = int, default = 15213)
parser.add_argument('--ckpt', type = int)
args = parser.parse_args()

# Locate model file and prepare directories for prediction and evaluation
expid = '%s-drop%.1f-batch%d-ckpt%d-%s-lr%.0e-pat%d-fac%.1f-seed%d' % (
    args.pooling,
    args.dropout,
    args.batch_size,
    args.ckpt_size,
    args.optimizer,
    args.init_lr,
    args.lr_patience,
    args.lr_factor,
    args.random_seed
)
WORKSPACE = os.path.join('../../workspace/dcase', expid)
MODEL_FILE = os.path.join(WORKSPACE, 'model', 'checkpoint%d.pt' % args.ckpt)
PRED_PATH = os.path.join(WORKSPACE, 'pred')
if not os.path.exists(PRED_PATH): os.makedirs(PRED_PATH)
PRED_FILE = os.path.join(PRED_PATH, 'checkpoint%d.mat' % args.ckpt)
EVAL_PATH = os.path.join(WORKSPACE, 'eval')
if not os.path.exists(EVAL_PATH): os.makedirs(EVAL_PATH)
EVAL_FILE = os.path.join(EVAL_PATH, 'checkpoint%d.txt' % args.ckpt)
with open(EVAL_FILE, 'w'):
    pass

def write_log(s):
    print s
    with open(EVAL_FILE, 'a') as f:
        f.write(s + '\n')

if os.path.exists(PRED_FILE):
    # Load saved predictions, no need to use GPU
    data = loadmat(PRED_FILE)
    thres = data['thres'].ravel()
    test_y = data['test_y']
    test_frame_y = data['test_frame_y']
    test_outputs = []
    test_outputs.append(data['test_global_prob'])
    test_outputs.append(data['test_frame_prob'])
    if args.pooling == 'att':
        test_outputs.append(data['test_frame_att'])
else:
    import torch
    import torch.nn as nn
    from torch.optim import *
    from torch.optim.lr_scheduler import *
    from torch.autograd import Variable
    from Net import Net
    from util_in import *

    # Load model
    model = Net(args).cuda()
    model.load_state_dict(torch.load(MODEL_FILE)['model'])
    model.eval()

    # Load data
    valid_x, valid_y, _ = bulk_load('DCASE_valid')
    test_x, test_y, test_hashes = bulk_load('DCASE_test')
    test_frame_y = load_dcase_test_frame_truth()

    # Predict
    valid_global_prob = model.predict(valid_x, verbose = False)
    thres = optimize_micro_avg_f1(valid_global_prob, valid_y)
    test_outputs = model.predict(test_x, verbose = True)

    # Save predictions
    data = {}
    data['thres'] = thres
    data['test_hashes'] = test_hashes
    data['test_y'] = test_y
    data['test_frame_y'] = test_frame_y
    data['test_global_prob'] = test_outputs[0]
    data['test_frame_prob'] = test_outputs[1]
    if args.pooling == 'att':
        data['test_frame_att'] = test_outputs[2]
    savemat(PRED_FILE, data)

# Evaluation
write_log('           ||          ||            Task A (recording level)           ||                       Task B (1-second segment level)                       ')
write_log('     CLASS ||    THRES ||   TP |   FN |   FP |  Prec. | Recall |     F1 ||   TP |   FN |   FP |  Prec. | Recall |     F1 |  Sub |  Del |  Ins |     ER ')
FORMAT1 = ' Micro Avg ||          || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f | %#4d | %#4d | %#4d | %6.02f '
FORMAT2 = ' %######9d || %8.0006f || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f |      |      |      |        '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT1)
write_log(SEP)

# test_y and test_frame_y are inconsistent in some places
# so when you evaluate Task A, use a "fake_test_frame_y" derived from test_y
fake_test_frame_y = numpy.tile(numpy.expand_dims(test_y, 1), (1, 100, 1))

# Micro-average performance across all classes
res_taskA = dcase_sed_eval(test_outputs, args.pooling, thres, fake_test_frame_y, 100, verbose = True)
res_taskB = dcase_sed_eval(test_outputs, args.pooling, thres, test_frame_y, 10, verbose = True)
write_log(FORMAT1 % (res_taskA.TP, res_taskA.FN, res_taskA.FP, res_taskA.precision, res_taskA.recall, res_taskA.F1,
                     res_taskB.TP, res_taskB.FN, res_taskB.FP, res_taskB.precision, res_taskB.recall, res_taskB.F1,
                     res_taskB.sub, res_taskB.dele, res_taskB.ins, res_taskB.ER))
write_log(SEP)

# Class-wise performance
N_CLASSES = test_outputs[0].shape[-1]
for i in range(N_CLASSES):
    outputs = [x[..., i:i+1] for x in test_outputs]
    res_taskA = dcase_sed_eval(outputs, args.pooling, thres[i], fake_test_frame_y[..., i:i+1], 100, verbose = True)
    res_taskB = dcase_sed_eval(outputs, args.pooling, thres[i], test_frame_y[..., i:i+1], 10, verbose = True)
    write_log(FORMAT2 % (i, thres[i],
                         res_taskA.TP, res_taskA.FN, res_taskA.FP, res_taskA.precision, res_taskA.recall, res_taskA.F1,
                         res_taskB.TP, res_taskB.FN, res_taskB.FP, res_taskB.precision, res_taskB.recall, res_taskB.F1))
