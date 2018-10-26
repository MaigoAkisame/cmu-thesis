import sys, os, os.path
import argparse
import numpy
from util_out import *
from util_f1 import *
from scipy.io import loadmat, savemat

# Parse input arguments
def mybool(s):
    return s.lower() in ['t', 'true', 'y', 'yes', '1']
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = 'ctl', choices = ['strong', 'mil', 'ctc', 'ctl', 'combine'])
parser.add_argument('--embedding_size', type = int, default = 512)
    # This is the embedding size after a pooling layer or after the GRU layer
    # After a non-pooling layer, the embeddings size will be twice this much
parser.add_argument('--n_conv_layers', type = int, default = 6)
parser.add_argument('--kernel_size', type = str, default = '3')     # 'n' or 'nxm'
parser.add_argument('--n_pool_layers', type = int, default = 6)
    # the pooling layers will be inserted uniformly into the conv layers
    # the should be at least 2 and at most 6 pooling layers
    # the first two pooling layers will have stride (2,2); later ones will have stride (1,2)
parser.add_argument('--max_concur', type = int, default = 1)
parser.add_argument('--mil_weight', type = float, default = 3.3)
parser.add_argument('--ctl_weight', type = float, default = 1.0)
parser.add_argument('--batch_norm', type = mybool, default = True)
parser.add_argument('--dropout', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 500)
parser.add_argument('--ckpt_size', type = int, default = 200)       # how many batches per checkpoint
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd'])
parser.add_argument('--init_lr', type = float, default = 1e-3)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 1.0)
parser.add_argument('--random_seed', type = int, default = 15213)
parser.add_argument('--ckpt', type = int)
args = parser.parse_args()
if 'x' not in args.kernel_size:
    args.kernel_size = args.kernel_size + 'x' + args.kernel_size

# Locate model file and prepare directories for prediction and evaluation
expid = '%s-embed%d-%dC%dP-kernel%s%s%s-%s-drop%.1f-batch%d-ckpt%d-%s-lr%.0e-pat%d-fac%.1f-seed%d' % (
    args.mode,
    args.embedding_size,
    args.n_conv_layers,
    args.n_pool_layers,
    args.kernel_size,
    '-concur%d' % args.max_concur if args.mode in ['ctl', 'combine'] else '',
    '-weight%g:%g' % (args.mil_weight, args.ctl_weight) if args.mode == 'combine' else '',
    'bn' if args.batch_norm else 'nobn',
    args.dropout,
    args.batch_size,
    args.ckpt_size,
    args.optimizer,
    args.init_lr,
    args.lr_patience,
    args.lr_factor,
    args.random_seed
)
WORKSPACE = os.path.join('../../workspace/sequential', expid)
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
    eval_frame_y = data['eval_frame_y']
    eval_frame_prob = data['eval_frame_prob']
else:
    import torch
    import torch.nn as nn
    from torch.optim import *
    from torch.optim.lr_scheduler import *
    from torch.autograd import Variable
    from Net import Net
    from util_in import *

    # Load model
    args.kernel_size = tuple(int(x) for x in args.kernel_size.split('x'))
    model = Net(args).cuda()
    model.load_state_dict(torch.load(MODEL_FILE)['model'])
    model.eval()

    # Load data
    valid_x, valid_frame_y, _, _ = bulk_load('GAS_valid')
    eval_x, eval_frame_y, _, eval_hashes = bulk_load('GAS_eval')

    # Predict
    if args.mode == 'ctc':
        thres = numpy.array([0.5] * eval_frame_y.shape[-1])
        eval_log_prob = model.predict(eval_x)
        eval_frame_prob = ctc_decode(eval_log_prob).astype('float32')
    else:
        valid_frame_prob = model.predict(valid_x)
        thres, valid_f1 = optimize_gas_valid(valid_frame_prob, valid_frame_y)
        eval_frame_prob = model.predict(eval_x)

    # Save predictions
    data = {}
    data['thres'] = thres
    data['eval_hashes'] = eval_hashes
    data['eval_frame_y'] = eval_frame_y
    data['eval_frame_prob'] = eval_frame_prob
    if args.mode == 'ctc':
        data['eval_log_prob'] = eval_log_prob
    savemat(PRED_FILE, data)

# Evaluation
write_log('     CLASS ||    THRES ||    TP |    FN |    FP |  Prec. | Recall |     F1 ')
FORMAT1 = ' Macro Avg ||          ||       |       |       |        |        | %6.02f '
FORMAT2 = ' %######9d || %8.0006f || %##5d | %##5d | %##5d | %6.02f | %6.02f | %6.02f '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT1)
write_log(SEP)

TP, FN, FP, precision, recall, f1 = evaluate_gas_eval(eval_frame_prob, thres, eval_frame_y, verbose = True)
write_log(FORMAT1 % f1.mean())
write_log(SEP)
N_CLASSES = len(f1)
for i in range(N_CLASSES):
    write_log(FORMAT2 % (i, thres[i], TP[i], FN[i], FP[i], precision[i], recall[i], f1[i]))
