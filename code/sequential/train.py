import sys, os, os.path, time
import argparse
import numpy
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from Net import Net
from ctc import ctc_loss
from ctl import ctl_loss
from util_in import *
from util_out import *
from util_f1 import *

torch.backends.cudnn.benchmark = True

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
parser.add_argument('--max_concur', type = int, default = 1)        # for mode == 'ctl' or 'combine' only
parser.add_argument('--mil_weight', type = float, default = 3.3)    # for mode == 'combine' only
parser.add_argument('--ctl_weight', type = float, default = 1.0)    # for mode == 'combine' only
parser.add_argument('--batch_norm', type = mybool, default = True)
parser.add_argument('--dropout', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 500)
parser.add_argument('--ckpt_size', type = int, default = 200)       # how many batches per checkpoint
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd'])
parser.add_argument('--init_lr', type = float, default = 1e-3)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 1.0)
parser.add_argument('--max_ckpt', type = int, default = 100)
parser.add_argument('--random_seed', type = int, default = 15213)
args = parser.parse_args()
if 'x' not in args.kernel_size:
    args.kernel_size = args.kernel_size + 'x' + args.kernel_size

numpy.random.seed(args.random_seed)

# Prepare log file and model directory
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
MODEL_PATH = os.path.join(WORKSPACE, 'model')
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
LOG_FILE = os.path.join(WORKSPACE, 'train.log')
with open(LOG_FILE, 'w'):
    pass

def write_log(s):
    timestamp = time.strftime('%m-%d %H:%M:%S')
    msg = '[' + timestamp + '] ' + s
    print msg
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

# Load data
write_log('Loading data ...')
train_gen = batch_generator(batch_size = args.batch_size, random_seed = args.random_seed)
gas_valid_x, gas_valid_y_frame, gas_valid_y_seq, _ = bulk_load('GAS_valid')
gas_eval_x, gas_eval_y_frame, gas_eval_y_seq, _ = bulk_load('GAS_eval')

# Build model
args.kernel_size = tuple(int(x) for x in args.kernel_size.split('x'))
model = Net(args).cuda()
if args.optimizer == 'sgd':
    optimizer = SGD(model.parameters(), lr = args.init_lr, momentum = 0.9, nesterov = True)
elif args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr = args.init_lr)
scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = args.lr_factor, patience = args.lr_patience) if args.lr_factor < 1.0 else None

# Train model
write_log('Training model ...')
write_log(' CKPT |    LR    |  Tr.LOSS || G.Val.F1 |  G.Ev.F1 ')
FORMAT  = ' %#4d | %8.0003g | %8.0006f || %8.0002f | %8.0002f '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
write_log(SEP)

checkpoint = 0
best_gv_f1 = None
best_ge_f1 = None

bce_loss = nn.BCELoss()
for checkpoint in range(1, args.max_ckpt + 1):
    # Train for args.ckpt_size batches
    model.train()
    train_loss = 0
    for batch in range(1, args.ckpt_size + 1):
        x, y_global, y_seq, y_frame = next(train_gen)
        optimizer.zero_grad()
        if args.mode == 'strong':
            frame_prob = model(x)
            loss = bce_loss(frame_prob, y_frame)
        elif args.mode == 'mil':
            frame_prob = model(x)
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)  # linear softmax pooling function
            loss = bce_loss(global_prob, y_global)
        elif args.mode == 'ctc':
            log_prob = model(x)
            seq_len = numpy.array([log_prob.shape[1]] * log_prob.shape[0])                  # actually all batches are the same size
            loss = ctc_loss(log_prob, seq_len, y_seq)
        elif args.mode == 'ctl':
            frame_prob = model(x)
            seq_len = numpy.array([frame_prob.shape[1]] * frame_prob.shape[0])              # actually all batches are the same size
            loss = ctl_loss(frame_prob, seq_len, y_seq, args.max_concur)
        elif args.mode == 'combine':
            frame_prob = model(x)
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)  # linear softmax pooling function
            mil_loss = bce_loss(global_prob, y_global)
            seq_len = numpy.array([frame_prob.shape[1]] * frame_prob.shape[0])              # actually all batches are the same size
            ctl_loss_ = ctl_loss(frame_prob, seq_len, y_seq, args.max_concur)
            loss = mil_loss * args.mil_weight + ctl_loss_ * args.ctl_weight
        train_loss += loss.data[0]
        if numpy.isnan(train_loss) or numpy.isinf(train_loss): break
        loss.backward()
        optimizer.step()
        sys.stderr.write('Checkpoint %d, Batch %d / %d, avg train loss = %f\r' % \
                         (checkpoint, batch, args.ckpt_size, train_loss / batch))
    train_loss /= args.ckpt_size

    # Evaluate model
    model.eval()
    def predict(x):
        if args.mode != 'ctc':
            return model.predict(x)
        else:
            log_prob = model.predict(x)
            return ctc_decode(log_prob).astype('float32')
    sys.stderr.write('Evaluating model on GAS_VALID ...\r')
    frame_prob = predict(gas_valid_x)
    thres, gv_f1 = optimize_gas_valid(frame_prob, gas_valid_y_frame)
    sys.stderr.write('Evaluating model on GAS_EVAL ...\r')
    frame_prob = predict(gas_eval_x)
    ge_f1 = evaluate_gas_eval(frame_prob, thres, gas_eval_y_frame, verbose = False)

    # Write log
    write_log(FORMAT % (checkpoint, optimizer.param_groups[0]['lr'], train_loss, gv_f1, ge_f1))

    # Abort if training has gone mad
    if numpy.isnan(train_loss) or numpy.isinf(train_loss):
        write_log('Aborted.')
        break

    # Save model regularly. Too bad I can't save the scheduler
    MODEL_FILE = os.path.join(MODEL_PATH, 'checkpoint%d.pt' % checkpoint)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    sys.stderr.write('Saving model to %s ...\r' % MODEL_FILE)
    torch.save(state, MODEL_FILE)

    # Update learning rate
    if scheduler is not None:
        scheduler.step(gv_f1)

    # Update best results
    if best_gv_f1 is None or gv_f1 > best_gv_f1:
        best_gv_f1 = gv_f1
        best_gv_ckpt = checkpoint
    if best_ge_f1 is None or ge_f1 > best_ge_f1:
        best_ge_f1 = ge_f1
        best_ge_ckpt = checkpoint

write_log('DONE!')
