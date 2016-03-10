#!/bin/bash

# tree
tree-conv=/gfs/songmeixu/tools/kaldi-tree-conv/tree-conv
tree-conv exp/tri1/tree data/lang/phones.txt exp/tri1/questions.int exp/tri1/htk_tree.txt

# hmm
convert2htk data/lang/phones.txt exp/tri1/final.mdl exp/tri1/treeacc exp/tri1/final.occs exp/tri1/tree > exp/tri1/hmmdefs

wait
echo "Done"
exit 0
