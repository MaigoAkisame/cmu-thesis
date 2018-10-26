import numpy

def dcase_sed_eval(outputs, pooling, thres, truth, seg_len, verbose = False):
    pred = outputs[1].reshape((-1, seg_len, outputs[1].shape[-1]))
    if pooling == 'max':
        seg_prob = pred.max(axis = 1)
    elif pooling == 'ave':
        seg_prob = pred.mean(axis = 1)
    elif pooling == 'lin':
        seg_prob = (pred * pred).sum(axis = 1) / pred.sum(axis = 1)
    elif pooling == 'exp':
        seg_prob = (pred * numpy.exp(pred)).sum(axis = 1) / numpy.exp(pred).sum(axis = 1)
    elif pooling == 'att':
        att = outputs[2].reshape((-1, seg_len, outputs[2].shape[-1]))
        seg_prob = (pred * att).sum(axis = 1) / att.sum(axis = 1)

    pred = seg_prob >= thres
    truth = truth.reshape((-1, seg_len, truth.shape[-1])).max(axis = 1)

    if not verbose:
        Ntrue = truth.sum(axis = 1)
        Npred = pred.sum(axis = 1)
        Ncorr = (truth & pred).sum(axis = 1)
        Nmiss = Ntrue - Ncorr
        Nfa = Npred - Ncorr

        error_rate = 1.0 * numpy.maximum(Nmiss, Nfa).sum() / Ntrue.sum()
        f1 = 2.0 * Ncorr.sum() / (Ntrue + Npred).sum()
        return error_rate, f1
    else:
        class Object(object):
            pass
        res = Object()
        res.TP = (truth & pred).sum()
        res.FN = (truth & ~pred).sum()
        res.FP = (~truth & pred).sum()
        res.precision = 100.0 * res.TP / (res.TP + res.FP)
        res.recall = 100.0 * res.TP / (res.TP + res.FN)
        res.F1 = 200.0 * res.TP / (2 * res.TP + res.FP + res.FN)
        res.sub = numpy.minimum((truth & ~pred).sum(axis = 1), (~truth & pred).sum(axis = 1)).sum()
        res.dele = res.FN - res.sub
        res.ins = res.FP - res.sub
        res.ER = 100.0 * (res.sub + res.dele + res.ins) / (res.TP + res.FN)
        return res
