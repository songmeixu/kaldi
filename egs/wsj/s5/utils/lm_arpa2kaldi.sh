#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# The output of this script is the symbol tables data/{words.txt,phones.txt},
# and the grammars and lexicons data/{L,G}{,_disambig}.fst

# To be run from ..
if [ -f path.sh ]; then . path.sh; fi

## compile G.fst from sri format arpa
lmfile=$1
workdir=$2
[ -z "$lmfile" ] && echo "you must input a LM file name" && exit 1;
test ! -f $lmfile && echo "no such LM file" && exit 1;

mkdir -p $workdir

cat $lmfile | \
  scripts/find_arpa_oovs.pl $workdir/words.txt  > $workdir/oovs_lm.txt

arpa2fst --natural-base=true $lmfile | fstprint | \
  scripts/remove_oovs.pl $workdir/oovs_lm.txt | \
  scripts/eps2disambig.pl | scripts/s2eps.pl | fstcompile --isymbols=$workdir/words.txt \
    --osymbols=$workdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
   fstrmepsilon > $workdir/G.fst

echo "Succeeded in formatting data."

## Checking that disambiguated lexicon times G is determinizable
fsttablecompose $workdir/L_disambig.fst $workdir/G.fst | fstdeterminize >/dev/null || echo Error
