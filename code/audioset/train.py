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
def mybool(s):
    return s.lower() in ['t', 'true', 'y', 'yes', '1']
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type = int, default = 1024) # this is the embedding size after a pooling layer
                                                                    # after a non-pooling layer, the embeddings size will be twice this much
parser.add_argument('--n_conv_layers', type = int, default = 10)
parser.add_argument('--kernel_size', type = str, default = '3')     # 'n' or 'nxm'
parser.add_argument('--n_pool_layers', type = int, default = 5)     # the pooling layers will be inserted uniformly into the conv layers
                                                                    # the should be at least 2 and at most 6 pooling layers
                                                                    # the first two pooling layers will have stride (2,2); later ones will have stride (1,2)
parser.add_argument('--batch_norm', type = mybool, default = True)
parser.add_argument('--dropout', type = float, default = 0.0)
parser.add_argument('--pooling', type = str, default = 'lin', choices = ['max', 'ave', 'lin', 'exp', 'att'])
parser.add_argument('--batch_size', type = int, default = 250)
parser.add_argument('--ckpt_size', type = int, default = 1000)      # how many batches per checkpoint
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd'])
parser.add_argument('--init_lr', type = float, default = 1e-3)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 0.8)
parser.add_argument('--max_ckpt', type = int, default = 30)
parser.add_argument('--random_seed', type = int, default = 15213)
args = parser.parse_args()
if 'x' not in args.kernel_size:
    args.kernel_size = args.kernel_size + 'x' + args.kernel_size

numpy.random.seed(args.random_seed)

# Prepare log file and model directory
expid = 'embed%d-%dC%dP-kernel%s-%s-drop%.1f-%s-batch%d-ckpt%d-%s-lr%.0e-pat%d-fac%.1f-seed%d' % (
    args.embedding_size,
    args.n_conv_layers,
    args.n_pool_layers,
    args.kernel_size,
    'bn' if args.batch_norm else 'nobn',
    args.dropout,
    args.pooling,
    args.batch_size,
    args.ckpt_size,
    args.optimizer,
    args.init_lr,
    args.lr_patience,
    args.lr_factor,
    args.random_seed
)
WORKSPACE = os.path.join('../../workspace/audioset', expid)
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
gas_valid_x, gas_valid_y, _ = bulk_load('GAS_valid')
gas_eval_x, gas_eval_y, _ = bulk_load('GAS_eval')
dcase_valid_x, dcase_valid_y, _ = bulk_load('DCASE_valid')
dcase_test_x, dcase_test_y, _ = bulk_load('DCASE_test')
dcase_test_frame_truth = load_dcase_test_frame_truth()
DCASE_CLASS_IDS = [318, 324, 341, 321, 307, 310, 314, 397, 325, 326, 323, 319, 14, 342, 329, 331, 316]

# Build model
args.kernel_size = tuple(int(x) for x in args.kernel_size.split('x'))
model = Net(args).cuda()
if args.optimizer == 'sgd':
    optimizer = SGD(model.parameters(), lr = args.init_lr, momentum = 0.9, nesterov = True)
elif args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr = args.init_lr)
scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = args.lr_factor, patience = args.lr_patience) if args.lr_factor < 1.0 else None
criterion = nn.BCELoss()

# Train model
write_log('Training model ...')
write_log('                            ||       GAS_VALID       ||        GAS_EVAL       || D_VAL ||              DCASE_TEST               ')
write_log(" CKPT |    LR    |  Tr.LOSS ||  MAP  |  MAUC |   d'  ||  MAP  |  MAUC |   d'  || Gl.F1 || Gl.F1 | Fr.ER | Fr.F1 | 1s.ER | 1s.F1 ")
FORMAT  = ' %#4d | %8.0003g | %8.0006f || %5.3f | %5.3f |%6.03f || %5.3f | %5.3f |%6.03f || %5.3f || %5.3f | %5.3f | %5.3f | %5.3f | %5.3f '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
write_log(SEP)

for checkpoint in range(1, args.max_ckpt + 1):
    # Train for args.ckpt_size batches
    model.train()
    train_loss = 0
    for batch in range(1, args.ckpt_size + 1):
        x, y = next(train_gen)
        optimizer.zero_grad()
        global_prob = model(x)[0]
        global_prob.clamp_(min = 1e-7, max = 1 - 1e-7)
        loss = criterion(global_prob, y)
        train_loss += loss.data[0]
        if numpy.isnan(train_loss) or numpy.isinf(train_loss): break
        loss.backward()
        optimizer.step()
        sys.stderr.write('Checkpoint %d, Batch %d / %d, avg train loss = %f\r' % \
                         (checkpoint, batch, args.ckpt_size, train_loss / batch))
        del x, y, global_prob, loss         # This line and next line: to save GPU memory
        torch.cuda.empty_cache()            # I don't know if they're useful or not
    train_loss /= args.ckpt_size

    # Evaluate model
    model.eval()
    sys.stderr.write('Evaluating model on GAS_VALID ...\r')
    global_prob = model.predict(gas_valid_x, verbose = False)
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob, gas_valid_y)
    sys.stderr.write('Evaluating model on GAS_EVAL ... \r')
    global_prob = model.predict(gas_eval_x, verbose = False)
    ge_map, ge_mauc, ge_dprime = gas_eval(global_prob, gas_eval_y)
    sys.stderr.write('Evaluating model on DCASE_VALID ...\r')
    global_prob = model.predict(dcase_valid_x, verbose = False)[:, DCASE_CLASS_IDS]
    thres = optimize_micro_avg_f1(global_prob, dcase_valid_y)
    dv_f1 = f1(global_prob >= thres, dcase_valid_y)
    sys.stderr.write('Evaluating model on DCASE_TEST ... \r')
    outputs = model.predict(dcase_test_x, verbose = True)
    outputs = tuple(x[..., DCASE_CLASS_IDS] for x in outputs)
    dt_f1 = f1(outputs[0] >= thres, dcase_test_y)
    dt_frame_er, dt_frame_f1 = dcase_sed_eval(outputs, args.pooling, thres, dcase_test_frame_truth, 1)
    dt_1s_er, dt_1s_f1 = dcase_sed_eval(outputs, args.pooling, thres, dcase_test_frame_truth, 10)

    # Write log
    write_log(FORMAT % (
        checkpoint, optimizer.param_groups[0]['lr'], train_loss,
        gv_map, gv_mauc, gv_dprime,
        ge_map, ge_mauc, ge_dprime,
        dv_f1, dt_f1, dt_frame_er, dt_frame_f1, dt_1s_er, dt_1s_f1
    ))

    # Abort if training has gone mad
    if numpy.isnan(train_loss) or numpy.isinf(train_loss):
        write_log('Aborted.')
        break

    # Save model. Too bad I can't save the scheduler
    MODEL_FILE = os.path.join(MODEL_PATH, 'checkpoint%d.pt' % checkpoint)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    sys.stderr.write('Saving model to %s ...\r' % MODEL_FILE)
    torch.save(state, MODEL_FILE)

    # Update learning rate
    if scheduler is not None:
        scheduler.step(gv_map)

write_log('DONE!')
