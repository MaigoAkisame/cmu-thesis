import numpy

# Compute F1 given predictions and truth
def f1(pred, truth):
    return 2.0 * (pred & truth).sum() / (pred.sum() + truth.sum())

# Given scores and truth for a single class (as 1-D numpy arrays), find optimal threshold and corresponding F1
# Statistics of other classes may be given to optimize micro-average F1
def optimize_f1(scores, truth, extraNcorr = 0, extraNtrue = 0, extraNpred = 0):
    # Start with predicting everything as negative
    best_thres = numpy.inf
    best_f1 = 0.0
    num = extraNcorr                                # number of correctly predicted instances
    den = extraNtrue + extraNpred + truth.sum()     # number of predicted instances + true instances
    instances = [(-numpy.inf, False)] + sorted(zip(scores, truth))
    # Lower the threshold gradually
    for i in range(len(instances) - 1, 0, -1):
        if instances[i][1]: num += 1
        den += 1
        if instances[i][0] > instances[i-1][0]:     # Can put threshold here
            f1 = 2.0 * num / den
            if f1 > best_f1:
                best_thres = (instances[i][0] + instances[i-1][0]) / 2
                best_f1 = f1
    return best_thres, best_f1

# Given scores and truth for many classes (as 2-D numpy arrays),
# find the optimal class-specific thresholds (as a 1-D numpy array) that maximizes the micro-average F1
# The algorithm is stochastic, but I have always observed deterministic results
def optimize_micro_avg_f1(scores, truth):
    # First optimize each class individually
    nClasses = truth.shape[1]
    thres = numpy.zeros(nClasses, dtype = 'float64')
    for i in range(nClasses):
        thres[i], _ = optimize_f1(scores[:,i], truth[:,i])
    Ntrue = truth.sum(axis = 0)
    Npred = (scores >= thres).sum(axis = 0)
    Ncorr = ((scores >= thres) & truth).sum(axis = 0)

    # Repeatly re-tune the threshold for each class until convergence
    candidates = range(nClasses)
    while len(candidates) > 0:
        i = numpy.random.choice(candidates)
        candidates.remove(i)
        old_thres = thres[i]
        thres[i], _ = optimize_f1(
            scores[:,i],
            truth[:,i],
            extraNcorr = Ncorr.sum() - Ncorr[i],
            extraNtrue = Ntrue.sum() - Ntrue[i],
            extraNpred = Npred.sum() - Npred[i],
        )
        if thres[i] != old_thres:
            Npred[i] = (scores[:,i] >= thres[i]).sum(axis = 0)
            Ncorr[i] = ((scores[:,i] >= thres[i]) & truth[:,i]).sum(axis = 0)
            candidates = range(nClasses)
            candidates.remove(i)

    return thres
