#!/usr/bin/env python3
# coding=utf-8
# Author: 2018 Meixu Song <songmeixu@outlook.com>
# License:
"""
This is what
Usage: check "--help"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse


def main(_):
  with open(FLAGS.in_spk2utt, "rt") as in_spk, open(FLAGS.keep_spk2utt, 'wt') as keep_spk, \
    open(FLAGS.out_spk2utt, 'wt') as out_spk:
    for line in in_spk:
      items = line.strip().split()
      spk = items[0]
      utts = items[1:]
      keep_utts = utts[:-FLAGS.utts]
      out_utts = utts[-FLAGS.utts:]
      print(spk, " ".join(keep_utts), file=keep_spk)
      print(spk, " ".join(out_utts), file=out_spk)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="split spk2utt by choose specific \
    number of utts from each speaker")
  parser.add_argument('-i', '--in', metavar='in-spk2utt', required=True,
                      dest='in_spk2utt', action='store',
                      help='input of spk2tt')
  parser.add_argument('-u', '--utts', metavar='utts', required=True,
                      dest='utts', action='store', type=int, default=None,
                      help='num of utts to choose from each speaker')
  parser.add_argument('--keep', metavar='keep-spk2utt', required=True,
                      dest='keep_spk2utt', action='store',
                      help='keep of spk2tt')
  parser.add_argument('--out', metavar='out-spk2utt', required=True,
                      dest='out_spk2utt', action='store',
                      help='output of spk2tt')
  FLAGS, unparsed = parser.parse_known_args()

  main(unparsed)
