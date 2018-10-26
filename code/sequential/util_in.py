import sys, os, os.path, glob
import cPickle
from scipy.io import loadmat
import numpy
from multiprocessing import Process, Queue
import torch
from torch.autograd import Variable

N_CLASSES = 35
N_WORKERS = 6

FEATURE_DIR = '../../data/sequential'
with open(os.path.join(FEATURE_DIR, 'normalizer.pkl'), 'rb') as f:
    mu, sigma = cPickle.load(f)

def sample_generator(file_list, random_seed = 15213):
    rng = numpy.random.RandomState(random_seed)
    while True:
        rng.shuffle(file_list)
        for filename in file_list:
            data = loadmat(filename)
            feat = ((data['feat'] - mu) / sigma).astype('float32')
            labels = data['labels'].astype('bool')
            for i in range(len(data['feat'])):
                yield feat[i], labels[i]

def worker(queues, file_lists, random_seed):
    generators = [sample_generator(file_lists[i], random_seed + i) for i in range(len(file_lists))]
    while True:
        for gen, q in zip(generators, queues):
            q.put(next(gen))

def batch_generator(batch_size, random_seed = 15213):
    queues = [Queue(5) for class_id in range(N_CLASSES)]
    file_lists = [sorted(glob.glob(os.path.join(FEATURE_DIR, 'GAS_train_unbalanced_class%02d_part*.mat' % class_id))) for class_id in range(N_CLASSES)]

    for worker_id in range(N_WORKERS):
        p = Process(target = worker, args = (queues[worker_id::N_WORKERS], file_lists[worker_id::N_WORKERS], random_seed))
        p.daemon = True
        p.start()

    rng = numpy.random.RandomState(random_seed)
    batch_x = []; batch_y_global = []; batch_y_seq = []; batch_y_frame = []
    while True:
        rng.shuffle(queues)
        for q in queues:
            x, y_frame = q.get()
            batch_x.append(x)
            batch_y_global.append(y_frame.max(axis = -2))
            batch_y_seq.append(mask2ctc(y_frame))
            batch_y_frame.append(y_frame)
            if len(batch_x) == batch_size:
                yield Variable(torch.from_numpy(numpy.stack(batch_x))).cuda(), \
                      Variable(torch.from_numpy(numpy.stack(batch_y_global).astype('float32'))).cuda(), \
                      batch_y_seq, \
                      Variable(torch.from_numpy(numpy.stack(batch_y_frame).astype('float32'))).cuda()
                batch_x = []; batch_y_global = []; batch_y_seq = []; batch_y_frame = []

def bulk_load(prefix):
    data = loadmat(os.path.join(FEATURE_DIR, prefix + '.mat'))
    x = ((data['feat'] - mu) / sigma).astype('float32')
    y_frame = data['labels'].astype('bool')
    y_seq = [mask2ctc(y) for y in y_frame]
    return x, y_frame, y_seq, data['hashes']

def mask2ctc(mask):
    z = numpy.zeros((1, mask.shape[-1]), dtype = 'bool')
    zp = numpy.concatenate([z, mask])
    pz = numpy.concatenate([mask, z])
    onset = (pz & ~zp).nonzero()
    offset = (zp & ~pz).nonzero()
    boundaries = sorted([(t, 1, event) for (t, event) in zip(*onset)] + [(t, -1, event) for (t, event) in zip(*offset)])    # time, onset/offset, event id
    return [bound[2] * 2 + {1:1, -1:2}[bound[1]] for bound in boundaries]
