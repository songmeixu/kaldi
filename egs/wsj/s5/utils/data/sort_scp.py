#!/usr/bin/env python
#-*- coding: utf8 -*-
"""
Author: 2018 Meixu Song <songmeixu_mega@icloud.com>

sort scp file by the key order in the first input file
cmd: python sort_scp.py key_file scp.in > scp.out
"""

import sys
from warnings import warn

key_file = sys.argv[1]
scp_in = sys.argv[2]

# load scp_in
k_v = {}
with open(scp_in, 'rt') as scp:
  for line in scp:
    line.strip()
    k, v = line.split()
    k_v[k] = v

with open(key_file, 'rt') as key:
  for line in key:
    line.strip()
    k = line.split()[0]

    if k in k_v:
      print(k, k_v[k])
    else:
      warn("Cannot find {} in input scp.".format(k))
