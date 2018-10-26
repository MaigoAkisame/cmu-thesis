import numpy
numpy.seterr(divide = 'ignore')
import torch
from torch.autograd import Variable

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def tensor(array):
    if array.dtype == 'bool':
        array = array.astype('uint8')
    return cuda(torch.from_numpy(array))

def variable(array):
    if isinstance(array, numpy.ndarray):
        array = tensor(array)
    return cuda(Variable(array))

def logsumexp(*args):
    M = reduce(torch.max, args)
    mask = M != -numpy.inf
    M[mask] += torch.log(sum(torch.exp(x[mask] - M[mask]) for x in args))
        # Must pick the valid part out, otherwise the gradient will contain NaNs
    return M

# Input arguments:
#   frameProb: a 3-D Variable of size N_SEQS * N_FRAMES * N_CLASSES containing the probability of each event at each frame.
#   seqLen: a list or numpy array indicating the number of valid frames in each sequence.
#   label: a list of label sequences.
# Note on implementation:
#   Anything that will be backpropped must be a Variable;
#   Anything used as an index must be a torch.cuda.LongTensor.
def ctl_loss(frameProb, seqLen, label, maxConcur = 1, debug = False):
    seqLen = numpy.array(seqLen)
    nSeqs, nFrames, nClasses = frameProb.size()

    # Clear the content in the frames of frameProb beyond seqLen
    frameIndex = numpy.tile(numpy.arange(nFrames), (nSeqs, 1))
    mask = variable(numpy.expand_dims(frameIndex < seqLen.reshape((nSeqs, 1)), 2))
    z = variable(torch.zeros(frameProb.size()))
    frameProb = torch.where(mask, frameProb, z)

    # Convert frameProb (probabilities of events) into probabilities of event boundaries
    z = variable(1e-7 * torch.ones((nSeqs, 1, nClasses)))       # Real zeros would cause NaNs in the gradient
    frameProb = torch.cat([z, frameProb, z], dim = 1)
    startProb = torch.clamp(frameProb[:, 1:] - frameProb[:, :-1], min = 1e-7)
    endProb = torch.clamp(frameProb[:, :-1] - frameProb[:, 1:], min = 1e-7)
    boundaryProb = torch.stack([startProb, endProb], dim = 3).view((nSeqs, nFrames + 1, nClasses * 2))

    blankLogProb = torch.log(1 - boundaryProb).sum(dim = 2)
        # blankLogProb[seq, frame] = log probability of emitting nothing at this frame
    deltaLogProb = torch.log(boundaryProb) - torch.log(1 - boundaryProb)
        # deltaLogProb[seq, frame, token] = log prob of emitting token minus log prob of not emitting token

    # Find out the lengths of the label sequences
    labelLen = tensor(numpy.array([len(x) for x in label]))

    # Put the label sequences into a Variable
    maxLabelLen = max(len(x) for x in label)
    L = numpy.zeros((nSeqs, maxLabelLen), dtype = 'int64')
    for i in range(nSeqs):
        L[i, :len(label[i])] = numpy.array(label[i]) - 1        # minus one because we no longer have a dedicated blank token
    label = tensor(L)

    if maxConcur > maxLabelLen:
        maxConcur = maxLabelLen

    # Compute alpha trellis
    # alpha[m, n] = log probability of having emitted n tokens in the m-th sequence at the current frame
    nStates = maxLabelLen + 1
    alpha = variable(-numpy.inf * torch.ones((nSeqs, nStates)))
    alpha[:, 0] = 0
    seqIndex = tensor(numpy.tile(numpy.arange(nSeqs), (nStates, 1)).T)
    dummyColumns = variable(-numpy.inf * torch.ones((nSeqs, maxConcur)))
    uttLogProb = variable(torch.zeros(nSeqs))
    for frame in range(nFrames + 1):        # +1 because we are considering boundaries
        # Case 0: don't emit anything at current frame
        p = alpha + blankLogProb[:, frame].view((-1, 1))
        alpha = p
        for i in range(1, maxConcur + 1):
            # Case i: emit i tokens at current frame
            p = p[:, :-1] + deltaLogProb[seqIndex[:, i:], frame, label[:, (i-1):]]
            alpha = logsumexp(alpha, torch.cat([dummyColumns[:, :i], p], dim = 1))
        # Collect probability for ends of utterances
        finishedSeqs = (seqLen == frame).nonzero()[0]
        if len(finishedSeqs) > 0:
            finishedSeqs = tensor(finishedSeqs)
            uttLogProb[finishedSeqs] = alpha[finishedSeqs, labelLen[finishedSeqs]].clone()

    # Return the per-frame negative log probability of all utterances (and per-utterance log probs if debug == True)
    loss = -uttLogProb.sum() / (seqLen + 1).sum()
    if debug:
        return loss, uttLogProb
    else:
        return loss

if __name__ == '__main__':
    def strip(variable):
        return variable.data.cpu().numpy()
    torch.set_printoptions(precision = 5)

    frameProb = numpy.array([[[0.1, 0.9, 0.9], [0.1, 0.9, 0.9], [0.1, 0.9, 0.9], [0.1, 0.9, 0.1]]], dtype = 'float32')  # event B all the time; event C in the first three frames
    frameProb = numpy.tile(frameProb, (4, 1, 1))
    frameProb = Variable(tensor(frameProb), requires_grad = True)
    label = [[3, 5, 6, 4], [3, 4], [5, 6], []]  # <B><C></C></B>; <B></B>; <C></C>; empty
    seqLen = numpy.array([4, 4, 4, 4])

    loss, uttLogProb = ctl_loss(frameProb, seqLen, label, maxConcur = 1, debug = True)
    print strip(loss), strip(torch.exp(uttLogProb))
    loss, uttLogProb = ctl_loss(frameProb, seqLen, label, maxConcur = 2, debug = True)
    print strip(loss), strip(torch.exp(uttLogProb))
    loss, uttLogProb = ctl_loss(frameProb, seqLen, label, maxConcur = 3, debug = True)
    print strip(loss), strip(torch.exp(uttLogProb))
    # Reference output:
    # [ 1.45882034] [  2.10689101e-03   2.61903927e-03   1.27433671e-03   3.03234774e-05]       # Prob of first label sequence is small
    # [ 1.26348567] [  1.04593262e-01   2.61992868e-03   1.27623521e-03   3.03234774e-05]       # Prob of first label sequence gets big, because <B><C> can be emitted at the same time
    # [ 1.263484  ] [  1.04596682e-01   2.61992868e-03   1.27623521e-03   3.03234774e-05]       # Prob of first label sequence stays almost the same, because it doesn't need to emit three tokens at the same time
    loss.backward()
    print frameProb.grad
