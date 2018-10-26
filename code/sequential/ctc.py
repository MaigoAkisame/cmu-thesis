import numpy
numpy.seterr(divide = 'ignore')
import torch
from torch.autograd import Variable

def logsumexp(*args):
    M = reduce(torch.max, args)
    mask = M != -numpy.inf
    M[mask] += torch.log(sum(torch.exp(x[mask] - M[mask]) for x in args))
        # Must pick the valid part out, otherwise the gradient will contain NaNs
    return M

# Input arguments:
#   logProb: a 3-D Variable of size N_SEQS * N_FRAMES * N_LABELS containing LOG probabilities.
#   seqLen: a list or numpy array indicating the number of valid frames in each sequence.
#   label: a list of label sequences.
# Note on implementation:
#   Anything that will be backpropped must be a Variable;
#   Anything used as an index must be a torch.cuda.LongTensor.
def ctc_loss(logProb, seqLen, label, debug = False):
    seqLen = numpy.array(seqLen)
    nSeqs, nFrames = logProb.size(0), logProb.size(1)

    # Find out the lengths of the label sequences
    labelLen = torch.from_numpy(numpy.array([len(x) for x in label])).cuda()

    # Insert blank symbol at the beginning, at the end, and between all symbols of the label sequences
    nStates = max(len(x) for x in label) * 2 + 1
    extendedLabel = numpy.zeros((nSeqs, nStates), dtype = 'int64')
    for i in range(nSeqs):
        extendedLabel[i, 1 : (len(label[i]) * 2) : 2] = label[i]
    label = torch.from_numpy(extendedLabel).cuda()

    # Compute alpha trellis
    dummyColumn = Variable(-numpy.inf * torch.ones((nSeqs, 1)).cuda())
    allSeqIndex = torch.from_numpy(numpy.tile(numpy.arange(nSeqs), (nStates, 1)).T).cuda()
    uttLogProb = Variable(torch.zeros(nSeqs).cuda())
    for frame in range(nFrames):
        if frame == 0:
            # Initialize the log probability first two states to log(1), and other states to log(0)
            alpha = Variable(-numpy.inf * torch.ones((nSeqs, nStates)).cuda())
            alpha[:, :2] = 0
        else:
            # Receive probability from previous frame
            p2 = alpha[:, :-2].clone()
            p2[label[:, 2:] == label[:, :-2]] = -numpy.inf
                # Probability can pass across labels two steps apart if they are different
            alpha = logsumexp(alpha,
                              torch.cat([dummyColumn, alpha[:, :-1]], 1),
                              torch.cat([dummyColumn, dummyColumn, p2], 1))
        # Multiply with the probability of current frame
        alpha += logProb[allSeqIndex, frame, label]
        # Collect probability for ends of utterances
        seqIndex = (seqLen == frame + 1).nonzero()[0]
        if len(seqIndex) > 0:
            seqIndex = torch.from_numpy(seqIndex).cuda()
            ll = labelLen[seqIndex]
            p = alpha[seqIndex, ll * 2].clone()
            if (ll > 0).any():
                p[ll > 0] = logsumexp(p[ll > 0], alpha[seqIndex[ll > 0], ll[ll > 0] * 2 - 1])
            uttLogProb[seqIndex] = p

    # Return the per-frame negative log probability of all utterances (and per-utterance log probs if debug == True)
    loss = -uttLogProb.sum() / seqLen.sum()
    if debug:
        return loss, uttLogProb
    else:
        return loss

if __name__ == '__main__':
    torch.set_printoptions(precision = 5)

    label = numpy.array([[2, 1, 1, 3],   # BAAC
                         [0, 0, 0, 0],   # null
                         [1, 0, 0, 0],   # A
                         [3, 2, 0, 0],   # CB
                         [0, 0, 0, 0],   # null
                         [1, 0, 0, 0],   # A
                         [3, 2, 0, 0]])  # CB
    seqLen = numpy.array([5, 3, 3, 3, 1, 1, 1])
    logProb = numpy.log(numpy.tile(numpy.array([[[0.1, 0.2, 0.3, 0.4]]], dtype = 'float32'), (len(seqLen), max(seqLen), 1)))
    logProb = Variable(torch.from_numpy(logProb).cuda(), requires_grad = True)
    loss, uttLogProb = ctc_loss(logProb, seqLen, label, debug = True)
    print loss, torch.exp(uttLogProb)
    # Expected output of torch.exp(uttLogProb): [0.00048, 0.001, 0.022, 0.12, 0.1, 0.2, 0]
    loss.backward()
#    print logProb.grad
