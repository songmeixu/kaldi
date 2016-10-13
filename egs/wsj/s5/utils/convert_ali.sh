#!/usr/bin/env bash

# Copyright 2016 Meixu Song
# Apache 2.0

. cmd.sh
. path.sh

alidir=$1
dir=$2

nj=`cat $alidir/num_jobs` || exit 1;
cmd=$train_cmd

# Convert the alignments.
echo "Converting alignments from $alidir to use current tree"
$cmd JOB=1:$nj $dir/log/convert.JOB.log \
  convert-ali $alidir/final.mdl $dir/final.mdl $dir/tree \
   "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;

wait
exit 0;
