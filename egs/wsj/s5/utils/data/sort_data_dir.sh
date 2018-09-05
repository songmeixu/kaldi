#!/usr/bin/env bash
# Author: Meixu Song (songmeixu_mega@icloud.com)
# Date: 2018-05-18

# This script sort data dir by frame lengths in ascending order.
# As to ctc training, it's easiler to converge when start from shorter utterances.

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: utils/data/sort_data_dir.sh <data-dir>"
  echo "e.g.: utils/data/sort_data_dir.sh data/train"
  echo "This script sort data dir by frame lengths in ascending order."
  echo "As to ctc training, it's easiler to converge when start from shorter utterances."
  exit 1
fi

data=$1
mkdir -p $data/.backup

[ ! -d $data ] && echo "$0: no such directory $data" && exit 1;

[ ! -f $data/feats.scp ] && echo "$0: no such file $data/feats.scp" && exit 1;

set -e -o pipefail -u

export LC_ALL=C

feat-to-len scp:$data/feats.scp ark,t:$data/tmp.len || exit 1;
sort -n -k2 $data/tmp.len >$data/feats.len
for x in utt2spk spk2utt feats.scp labels.scp text segments wav.scp cmvn.scp vad.scp \
    reco2file_and_channel spk2gender utt2lang utt2uniq utt2dur reco2dur utt2num_frames; do
  if [ -f $data/$x ]; then
    cp $data/$x $data/.backup/$x
    utils/data/sort_scp.py $data/feats.len $data/.backup/$x >$data/$x
  fi
done
rm $data/tmp.len

echo "sort_data_dir.sh: old files are kept in $data/.backup"
