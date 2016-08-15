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

#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Get transition prob in zeus hclg.trans format, for decoding usage\n"
        "Usage:  am-transitions <transition/model-file> <out-trans>\n"
        "e.g.: \n"
        " am-transitions in.mdl out.hclg.trans\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string transition_model_filename = po.GetArg(1);
    std::string fname = po.GetArg(2);

    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    std::ofstream trans_ofp(fname.c_str(), std::ios::out | std::ios::binary);

    float trans_prob;

    int32 pdf_num = trans_model.NumPdfs();
    int32 state_num = pdf_num + 1; // 0 for <eps>
    trans_ofp.write((const char *) &state_num, sizeof(int));

    Vector<BaseFloat> trans_probs;
    trans_probs.Resize(2*pdf_num);
    trans_probs.Set(1.0f);

    for (int t = 1; t <= trans_model.NumTransitionIds(); ++t) {
      int32 pdf_id = trans_model.TransitionIdToPdf(t);
      int32 trans_state = trans_model.TransitionIdToTransitionState(t);
      int32 trans_id;
      if (trans_probs(pdf_id) > 0.0f) {
        trans_id = trans_model.PairToTransitionId(trans_state, 0);
        trans_probs(pdf_id) = trans_model.GetTransitionLogProb(trans_id);
      }
      if (trans_probs(pdf_num + pdf_id) > 0.0f) {
        trans_id = trans_model.PairToTransitionId(trans_state, 1);
        trans_probs(pdf_num + pdf_id) = trans_model.GetTransitionLogProb(trans_id);
      }
    }

    // [0] for eps, compatible with qihoo .trans format
    // trans_out
    trans_prob = log(0.5);
    trans_ofp.write((const char *) &trans_prob, sizeof(float));
    for (int i = 0; i < pdf_num; i++) {
      trans_prob = trans_probs(i);
      trans_ofp.write((const char *) &trans_prob, sizeof(float));
    }
    // trans_self
    trans_ofp.write((const char *) &trans_prob, sizeof(float));
    for (int i = pdf_num; i < 2 * pdf_num; i++) {
      trans_prob = trans_probs(i);
      trans_ofp.write((const char *) &trans_prob, sizeof(float));
    }

    trans_ofp.close();

    KALDI_LOG << trans_probs;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

