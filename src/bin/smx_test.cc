//
// Created by songmeixu on 15/11/19.
//

// bin/show-transitions.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <sstream>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "gmm/am-diag-gmm.h"
#include "tree/build-tree.h"
#include "fst/fstlib.h"

using namespace kaldi;
using std::vector;
using kaldi::int32;

int main(int argc, const char *argv[]) {
  try {
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "convert kaldi hmm.mdl to htk mmf\n"
            "Usage:  smx_test <phones-symbol-table> <in-kaldi-mdl-file> <treeacc> <*.occs> <tree>\n"
            "e.g.: \n"
            " smx_test phones.txt final.mdl treeacc occs tree\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string phones_symtab_filename = po.GetArg(1),
        model_in_filename = po.GetArg(2),
        acc_filename = po.GetOptArg(3),
        occ_filename = po.GetOptArg(4),
        tree_filename = po.GetOptArg(5);

    // 1. load phones.txt
    fst::SymbolTable *phones_symtab = fst::SymbolTable::ReadText(phones_symtab_filename);
    if (!phones_symtab)
      KALDI_ERR << "Could not read symbol table from file " << phones_symtab_filename;

    // 2. construct AmDiagGmm & TransitionModel
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    // for tri+ trees, read the tree stats;  this gives us basically all
    // phones-in-context that may be linked to an individual model
    // (in practice, many of them will be shared, but we plot them anyways)

    // build-tree-questions.h:typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType
    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(acc_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
//    KALDI_LOG << "Number of separate statistics is " << stats.size();

    // typedef std::vector<std::pair<EventKeyType,EventValueType> > EventType

    // print hmm: ~h
    // read the tree, get all the leaves
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);
    // now, for each tree stats element, query the tree to get the pdf-id

    int32 trans_state;
    int32 trans_id;
    BaseFloat loop_prob;
    BaseFloat out_prob;
    std::vector<int32> phones_id;
    std::vector<int32> pdfs_id;

    phones_id.push_back(36);
    phones_id.push_back(19);
    phones_id.push_back(87);

    for (int32 i = 0; i < phones_id.size(); ++i) {
      int32 pdf_id;
      ctx_dep.Compute(phones_id, i, &pdf_id);
      pdfs_id.push_back(pdf_id);
    }
    KALDI_LOG << "pdfs_id: " << pdfs_id[0] << "-" << pdfs_id[1] << "+" << pdfs_id[2];

    trans_state = trans_model.TripleToTransitionState(19, 2, pdfs_id[2]);
    trans_id = trans_model.PairToTransitionId(trans_state, 0);
    loop_prob = trans_model.GetTransitionProb(trans_id);
    KALDI_LOG << " trans_id: " << trans_id << " " << loop_prob;
    trans_id = trans_model.PairToTransitionId(trans_state, 1);
    out_prob = trans_model.GetTransitionProb(trans_id);
    KALDI_LOG << " trans_id: " << trans_id << " " << out_prob;

    delete phones_symtab;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


