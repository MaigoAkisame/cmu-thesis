TALNet_FILE=../../workspace/audioset/TALNet/model/TALNet.pt
if ! [ -f $TALNet_FILE ]; then
  mkdir -p $(dirname $TALNet_FILE)
  wget -O $TALNet_FILE http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/model/TALNet.pt
fi
python eval.py --TALNet
