# cmu-thesis

This repository contains the code for three experiments in my PhD thesis, [Polyphonic Sound Event Detection with Weak Labeling](http://www.cs.cmu.edu/~yunwang/papers/cmu-thesis.pdf):

* Sound event detection with **presence/absence labeling** on the **[DCASE 2017 challenge](http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-large-scale-sound-event-detection)** (Chapter 3.2)
* Sound event detection with **presence/absence labeling** on **[Google Audio Set](https://research.google.com/audioset/)** (Chapter 3.3)
* Sound event detection with **sequential labeling** on a subset of **[Google Audio Set](https://research.google.com/audioset/)** (Chapter 4)

## Prerequisites

Hardware:
* A GPU
* Large storage (1 TB recommended)

Software:
* Python 2.7
* PyTorch (I used version 0.4.0a0+d3b6c5e)
* numpy, scipy, [joblib](https://pypi.org/project/joblib/)

## Quick Start

```python
# Clone the repository
git clone https://github.com/MaigoAkisame/cmu-thesis.git

# Download the data: may take up to 1 day!
cd cmu-thesis/data
./download.sh

# Train a model for the DCASE experiment using default settings
cd ../code/dcase
python train.py            # Needs to run on a GPU

# Evaluate the model at Checkpoint 25
python eval.py --ckpt=25   # Needs to run on a GPU for the first time

# Download and evaluate the TALNet model for the Audio Set experiment
cd ../audioset
./eval-TALNet.sh           # Needs to run on a GPU for the first time
```

## Organization of the Repository

### code

The `code` directory contains three sub-directories: `dcase`, `audioset`, and `sequential`. These contain the code for the three experiments. In each subdirectory:

* `Net.py` defines the network architecture (you don't need to execute this script directly);
* `train.py` trains the network;
* `eval.py` evaluates the network's performance.

The `train.py` and `eval.py` script can take many command line arguments, which specify the architecture of the network and the hyperparameters used during training. If you encounter "out of memory" errors, a good idea is to reduce the batch size.

Some scripts that may be of special interest:

* `code/*/util_in.py`: Implements data balancing so that each minibatch contains roughly equal numbers of recordings of each event type;
* `code/sequential/ctc.py`: My implementation of connectionist temporal classification (CTC);
* `code/sequential/ctl.py`: My implementation of connectionist temporal localization (CTL).

### data

The script `data/download.sh` will download and extract the following three archives in the `data` directory:

* [dcase.tgz](http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/data/dcase.tgz) (4.9 GB)
* [audioset.tgz](http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/data/dcase.tgz) (341 GB)
* [sequential.tgz](http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/data/dcase.tgz) (63 GB)

These archives contain Matlab data files (with the `.mat` extension) that store the filterbank features and ground truth labels. They can be loaded with the `scipy.io.loadmat` function in Python. Each Matlab file contains three matrices:

* `feat`: Filterbank features, a float32 array of shape (n, 400, 64) (n recordings, 400 frames, 64 frequency bins);
* `labels`:
  * Presence/absence labeling, a boolean array of shape (n, m) (n recordings, m event types), or
  * or strong labelng, a boolean array of shape (n, 100, m) (n recordings, 100 frames, m event types);
* `hashes`: A character array of size (n, 11), containing the YouTube hash IDs of the recordings.

Training recordings are organized by class (so data balancing can be done easily), and each Matlab file contains up to 101 recordings. Validation and test/evaluation recordings are stored in Matlab files that contain up to 500 recordings each.

Because the data is so huge, I do not provide the code for downloading the raw data, extracting features, and organizing the features and labels into Matlab data files. The whole process took me more than a month and endless babysitting!

### workspace

The training logs, trained models, predictions on the test/evaluation recordings, and evaluation results will be generated in this directory. The sub-directory names will reflect the network architecture and hyperparameters for training.

The script `code/audioset/eval-TALNet.py` will download the TALNet model and store it at `workspace/audioset/TALNet/model/TALNet.pt`. At the time of my graduation (October 2018), this is the best model that can both classify and localize sound events on Google Audio Set.

## Citing

If you use this code in your research, please cite my PhD thesis:

* Yun Wang, "Polyphonic sound event detection with weak labeling", PhD thesis, Carnegie Mellon University, Oct. 2018.

and/or the following publications:

* Yun Wang, Juncheng Li and Florian Metze, "A comparison of five multiple instance learning pooling functions for sound event detection with weak labeling," arXiv e-prints, Oct. 2018. [Online]. Available: <http://arxiv.org/abs/1810.09050>.
* Yun Wang and Florian Metze, "Connectionist temporal localization for sound event detection with sequential labeling," arXiv e-prints, Oct. 2018. [Online]. Available: <http://arxiv.org/abs/1810.09052>.
