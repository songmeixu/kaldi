#!/usr/bin/env bash

cmd=run.pl
echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

nj=$1
alidir=$2
dir=$3

# Convert the alignments.
echo "$0: converting alignments from $alidir"
$cmd JOB=1:$nj $dir/log/copy.JOB.log \
  copy-int-vector \
    "ark:gunzip -c $alidir/label.JOB.gz|" \
    "ark,t,scp:$dir/labels.JOB.ark,$dir/labels.JOB.scp" || exit 1;

echo "Done"

exit 0