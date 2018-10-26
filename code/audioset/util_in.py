import sys, os, os.path, glob
import cPickle
from scipy.io import loadmat
import numpy
from multiprocessing import Process, Queue
import torch
from torch.autograd import Variable

N_CLASSES = 527
N_WORKERS = 6

GAS_FEATURE_DIR = '../../data/audioset'
DCASE_FEATURE_DIR = '../../data/dcase'
with open(os.path.join(GAS_FEATURE_DIR, 'normalizer.pkl'), 'rb') as f:
    mu, sigma = cPickle.load(f)

def sample_generator(file_list, random_seed = 15213):
    rng = numpy.random.RandomState(random_seed)
    while True:
        rng.shuffle(file_list)
        for filename in file_list:
            data = loadmat(filename)
            feat = ((data['feat'] - mu) / sigma).astype('float32')
            labels = data['labels'].astype('float32')
            for i in range(len(data['feat'])):
                yield feat[i], labels[i]

def worker(queues, file_lists, random_seed):
    generators = [sample_generator(file_lists[i], random_seed + i) for i in range(len(file_lists))]
    while True:
        for gen, q in zip(generators, queues):
            q.put(next(gen))

def batch_generator(batch_size, random_seed = 15213):
    queues = [Queue(5) for class_id in range(N_CLASSES)]
    file_lists = [sorted(glob.glob(os.path.join(GAS_FEATURE_DIR, 'GAS_train_unbalanced_class%03d_part*.mat' % class_id))) for class_id in range(N_CLASSES)]

    for worker_id in range(N_WORKERS):
        p = Process(target = worker, args = (queues[worker_id::N_WORKERS], file_lists[worker_id::N_WORKERS], random_seed))
        p.daemon = True
        p.start()

    rng = numpy.random.RandomState(random_seed)
    batch = []
    while True:
        rng.shuffle(queues)
        for q in queues:
            batch.append(q.get())
            if len(batch) == batch_size:
                yield tuple(Variable(torch.from_numpy(numpy.stack(x))).cuda() for x in zip(*batch))
                batch = []

def bulk_load(prefix):
    feat = []; labels = []; hashes = []
    for filename in sorted(glob.glob(os.path.join(GAS_FEATURE_DIR, '%s_*.mat' % prefix)) +
                           glob.glob(os.path.join(DCASE_FEATURE_DIR, '%s_*.mat' % prefix))):
        data = loadmat(filename)
        feat.append(((data['feat'] - mu) / sigma).astype('float32'))
        labels.append(data['labels'].astype('bool'))
        hashes.append(data['hashes'])
    return numpy.concatenate(feat), numpy.concatenate(labels), numpy.concatenate(hashes)

def load_dcase_test_frame_truth():
    return cPickle.load(open(os.path.join(DCASE_FEATURE_DIR, 'DCASE_test_frame_label.pkl'), 'rb'))
