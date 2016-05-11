#!/bin/bash

gmm_dir=exp/dev/tri1
lang=data/lang

# tree
tree_conv=/glfs/songmeixu/tools/kaldi-tree-conv/tree-conv
$(tree_conv) $gmm_dir/tree $lang/phones.txt $gmm_dir/questions.int $gmm_dir/htk_tree.txt

# hmm
convert2htk $lang/phones.txt $gmm_dir/final.mdl $gmm_dir/treeacc $gmm_dir/final.occs $gmm_dir/tree > $gmm_dir/hmmdefs

wait
echo "Done"
exit 0
