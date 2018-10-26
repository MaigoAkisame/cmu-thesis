import numpy
from util_f1 import *
from joblib import Parallel, delayed

N_JOBS = 6

def ctc_decode(log_prob):
    # Decode log_prob (boundary probabilities, batch * frame * (2n+1)) to frame_pred (boolean event decisions, batch * frame * n)
    nSeqs, nFrames, nLabels = log_prob.shape
    nClasses = (nLabels - 1) / 2
    frame_pred = numpy.zeros((nSeqs, nFrames, nClasses), dtype = 'bool')
    for i in range(nSeqs):
        onset = [None] * nClasses
        prev_token = 0
        for t, token in zip(range(nFrames), log_prob[i].argmax(axis = 1)):
            if token == 0: continue
            if token % 2 == 1:      # onset of event
                event = (token - 1) / 2
                onset[event] = t
            else:                   # offset of event
                event = token / 2 - 1
                if onset[event] is not None:
                    frame_pred[i, onset[event] : t + 1, event] = True
                onset[event] = None
    return frame_pred

def optimize_gas_valid(pred, y):
    nClasses = y.shape[-1]
    result = Parallel(n_jobs = N_JOBS)(delayed(optimize_f1)(pred[..., i].ravel(), y[..., i].ravel()) for i in range(nClasses))
    thres = numpy.array([r[0] for r in result], dtype = 'float64')
    class_f1 = numpy.array([r[1] for r in result], dtype = 'float32') * 100.0
    return thres, class_f1.mean()

def TP_FN_FP(pred, truth):
    TP = (pred & truth).sum()
    FN = (~pred & truth).sum()
    FP = (pred & ~truth).sum()
    return (TP, FN, FP)

def evaluate_gas_eval(pred, thres, truth, verbose = False):
    # if verbose == False, return only the macro-average F1
    # if verbose == True, return the class-wise TP, FN, FP, precision, recall, F1
    pred = pred >= thres
    nClasses = len(thres)
    stats = Parallel(n_jobs = N_JOBS)(delayed(TP_FN_FP)(pred[..., i], truth[..., i]) for i in range(nClasses))
    TP, FN, FP = numpy.array(stats, dtype = 'int32').T
    f1 = 200.0 * TP / (2 * TP + FN + FP)
    if not verbose: return f1.mean()
    precision = 100.0 * TP / (TP + FP)
    recall = 100.0 * TP / (TP + FN)
    return TP, FN, FP, precision, recall, f1
