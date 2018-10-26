import sys, os, os.path, time
import argparse
import numpy
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from Net import Net
from util_in import *
from util_out import *
from util_f1 import *

torch.backends.cudnn.benchmark = True

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
parser.add_argument('--max_ckpt', type = int, default = 50)
parser.add_argument('--random_seed', type = int, default = 15213)
args = parser.parse_args()

numpy.random.seed(args.random_seed)

# Prepare log file and model directory
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
MODEL_PATH = os.path.join(WORKSPACE, 'model')
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
LOG_FILE = os.path.join(WORKSPACE, 'train.log')
with open(LOG_FILE, 'w'):
    pass

def write_log(s):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    msg = '[' + timestamp + '] ' + s
    print msg
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

# Load data
write_log('Loading data ...')
valid_x, valid_y, _ = bulk_load('DCASE_valid')
test_x, test_y, _ = bulk_load('DCASE_test')
test_frame_y = load_dcase_test_frame_truth()

# Build model
write_log('Building model ...')
model = Net(args).cuda()
if args.optimizer == 'sgd':
    optimizer = SGD(model.parameters(), lr = args.init_lr, momentum = 0.9, nesterov = True)
elif args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr = args.init_lr)
if args.lr_factor < 1.0:
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = args.lr_factor, patience = args.lr_patience)
criterion = nn.BCELoss()
def bce_loss(input, target):
    return -numpy.log(numpy.where(target, input, 1 - input)).sum() / input.size

# Train model
write_log('Training model ...')
write_log('                                       || D_VAL ||              DCASE_TEST               ')
write_log(' CKPT |    LR    |  Tr.LOSS | Val.LOSS || Gl.F1 || Gl.F1 | Fr.ER | Fr.F1 | 1s.ER | 1s.F1 ')
FORMAT  = ' %#4d | %8.0003g | %8.0006f | %8.0006f || %5.3f || %5.3f | %5.3f | %5.3f | %5.3f | %5.3f '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
write_log(SEP)

gen_train = batch_generator(args.batch_size, args.random_seed)
for ckpt in range(1, args.max_ckpt + 1):
    model.train()
    train_loss = 0
    for i in range(args.ckpt_size):
        x, y = next(gen_train)
        optimizer.zero_grad()
        global_prob = model(x)[0]
        global_prob.clamp_(min = 1e-7, max = 1 - 1e-7)
        loss = criterion(global_prob, y)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        sys.stderr.write('Checkpoint %d, Batch %d / %d, avg train loss = %f\r' % (ckpt, i + 1, args.ckpt_size, train_loss / (i + 1)))
    train_loss /= args.ckpt_size

    # Compute validation loss, validation F1 and test F1
    model.eval()
    valid_global_prob = model.predict(valid_x, verbose = False)
    valid_loss = bce_loss(valid_global_prob, valid_y)
    thres = optimize_micro_avg_f1(valid_global_prob, valid_y)
    valid_global_f1 = f1(valid_global_prob >= thres, valid_y)
    test_outputs = model.predict(test_x, verbose = True)
    test_global_f1 = f1(test_outputs[0] >= thres, test_y)
    test_frame_er, test_frame_f1 = dcase_sed_eval(test_outputs, args.pooling, thres, test_frame_y, 1)   # every 1 frame is a segment
    test_1s_er, test_1s_f1 = dcase_sed_eval(test_outputs, args.pooling, thres, test_frame_y, 10)        # every 10 frame is a segment

    # Write log
    write_log(FORMAT % (
        ckpt, optimizer.param_groups[0]['lr'], train_loss, valid_loss,
        valid_global_f1, test_global_f1, test_frame_er, test_frame_f1, test_1s_er, test_1s_f1
    ))

    # Abort if training has gone mad
    if numpy.isnan(train_loss) or numpy.isinf(train_loss):
        write_log('Aborted.')
        break

    # Save model. Too bad I can't save the scheduler
    MODEL_FILE = os.path.join(MODEL_PATH, 'checkpoint%d.pt' % ckpt)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, MODEL_FILE)

    # Update learning rate
    if args.lr_factor < 1.0:
        scheduler.step(valid_loss)

write_log('DONE!')
