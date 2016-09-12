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
//#include "matrix/kaldi-vector.h"
#include "hmm/transition-model.h"
#include "gmm/am-diag-gmm.h"
#include "tree/build-tree.h"
//#include "tree/clusterable-classes.h"
//#include "tree/context-dep.h"
//#include "tree/build-tree-questions.h"
#include "fst/fstlib.h"

using namespace kaldi;
using std::vector;
using kaldi::int32;

static bool is_chain_gmm = false;

std::string operator*(std::string const &s, size_t n) {
  std::string r;  // empty string
  r.reserve(n * s.size());
  for (size_t i=0; i<n; i++)
    r += s;
  return r;
}

// Generate a string representation of the given EventType;  the symtable is
// optional, so is the request for positional symbols (tri-phones: 0-left,
// 1-center, 2-right.
static std::string EventTypeToString(EventType &e,
                                     const fst::SymbolTable *phones_symtab,
                                     const TransitionModel &trans_model,
                                     const ContextDependency &ctx_dep) {
  // make sure it's sorted so that the kPdfClass is the first element!
  std::sort(e.begin(), e.end());

  // first plot the pdf-class

  std::vector<int32> phones_id;
  std::vector<int32> pdfs_id;
  std::stringstream ss;
  ss << "~h \"";
  for (size_t i = 1; i < e.size(); ++i) {
    if (i == 2)
      ss << "-";
    else if (i == 3)
      ss << "+";

    phones_id.push_back(e[i].second);
    if (e[i].second ==0) return ""; // skip <eps>
    std::string phn = phones_symtab->Find(static_cast<kaldi::int64>(e[i].second));
    if (phn.empty()) {
      // in case we can't resolve the symbol, plot the ID
      KALDI_WARN << "No phone found for ID " << e[i].second;
      ss << e[i].second;
    } else
      ss << phn;
  }
  ss << "\"" << std::endl;

  int center_phone_idx = e.size() / 2;
  HmmTopology::TopologyEntry phone_topo = trans_model.GetTopo().TopologyForPhone(phones_id[center_phone_idx]);
  int num_states = phone_topo.size();
  for (int32 i = 0; i < num_states; ++i) {
    int32 pdf_id;
    ctx_dep.Compute(phones_id, i, &pdf_id);
    pdfs_id.push_back(pdf_id);
  }

  ss << "<BEGINHMM>" << std::endl;
  ss << "<NUMSTATES> " << num_states + 2 << endl;
  for (size_t i = 0; i < phones_id.size(); ++i) {
    ss << "<STATE> " << i + 2 << std::endl;
    ss << "~s \"PDFID_" << pdfs_id[i] << "\"" << std::endl;
  }
  // print transion prob
  ss << "<TRANSP> " << phones_id.size() + 2 << std::endl;
  ss << " 0 1" << std::string(" 0") * num_states << std::endl;
  int32 trans_state;
  int32 trans_id;
  BaseFloat loop_prob;
  BaseFloat out_prob;
  for (int32 i = 0; i < num_states; ++ i) {
    trans_state = trans_model.TripleToTransitionState(phones_id[center_phone_idx], i, pdfs_id[i]);
    trans_id = trans_model.PairToTransitionId(trans_state, 0);
    loop_prob = trans_model.GetTransitionProb(trans_id);
    trans_id = trans_model.PairToTransitionId(trans_state, 1);
    out_prob = trans_model.GetTransitionProb(trans_id);
    if (is_chain_gmm && i == 1)
      ss << " 0 0" << loop_prob << " " << out_prob << std::endl;
    else
      ss << " 0" << std::string(" 0") * i << loop_prob << " " << out_prob << std::string(" 0") * (num_states - i - 1) << std::endl;
  }
  ss << std::string(" 0") * (num_states + 2) << std::endl;
  ss << "<ENDHMM>" << std::endl;

  return ss.str();
}

void PrintUnseen(std::ostream &os,
                 const TransitionModel &trans_model,
                 const fst::SymbolTable *phones_symtab,
                 const Vector<double> *occs,
                 const ContextDependency &ctx_dep) {
  KALDI_ASSERT(occs->Dim() == trans_model.NumPdfs());
  std::set<int32> unseen_phone;
  for (int pdf = 0; pdf < occs->Dim(); ++pdf) {
    if ((*occs)(pdf) != 0) {
      continue;
    }
    std::vector<int32> pdfs(1, pdf), phones;
    GetPhonesForPdfs(trans_model, pdfs, &phones);
    KALDI_ASSERT(phones.size() == 1);
    unseen_phone.insert(phones[0]);
  }

  for (std::set<int32>::iterator it = unseen_phone.begin(); it != unseen_phone.end(); ++it) {
    std::string phn = phones_symtab->Find(*it);
    std::vector<int32> triphone(3, 1);
    std::vector<int32> pdfs_id;
    triphone[1] = *it;
    HmmTopology::TopologyEntry phone_topo = trans_model.GetTopo().TopologyForPhone(*it);
    int num_states = phone_topo.size();
    for (int32 i = 0; i < num_states; ++i) {
      int32 pdf_id;
      ctx_dep.Compute(triphone, i, &pdf_id);
      pdfs_id.push_back(pdf_id);
    }
    os << "~h \"sil-" << phn << "+sil\"" << std::endl;
    os << "<BEGINHMM>" << std::endl;
    os << "<NUMSTATES> " << num_states + 2 << endl;
    for (size_t i = 0; i < num_states; ++i) {
      os << "<STATE> " << i + 2 << std::endl;
      os << "~s \"PDFID_" << pdfs_id[i] << "\"" << std::endl;
    }
    // print transion prob
    os << "<TRANSP> " << num_states + 2 << std::endl;
    os << " 0 1" << std::string(" 0") * num_states << std::endl;
    int32 trans_state;
    int32 trans_id;
    BaseFloat loop_prob;
    BaseFloat out_prob;
    for (int32 i = 0; i < num_states; ++ i) {
      trans_state = trans_model.TripleToTransitionState(*it, i, pdfs_id[i]);
      trans_id = trans_model.PairToTransitionId(trans_state, 0);
      loop_prob = trans_model.GetTransitionProb(trans_id);
      trans_id = trans_model.PairToTransitionId(trans_state, 1);
      out_prob = trans_model.GetTransitionProb(trans_id);
      if (is_chain_gmm && i == 1)
        os << " 0 0" << loop_prob << " " << out_prob << std::endl;
      else
        os << " 0" << std::string(" 0") * i << loop_prob << " " << out_prob << std::string(" 0") * (num_states - i - 1) << std::endl;
    }
    os << std::string(" 0") * (num_states + 2) << std::endl;
    os << "<ENDHMM>" << std::endl;
  }
}

int main(int argc, const char *argv[]) {
  try {
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "convert kaldi hmm.mdl to htk mmf\n"
            "Usage:  gmm2htk [option] <phones-symbol-table> <in-kaldi-mdl-file> <treeacc> <*.occs> <tree>\n"
            "e.g.: \n"
            "gmm2htk phones.txt final.mdl treeacc occs tree > hmmdefs\n";

    bool chain = false;
    ParseOptions po(usage);
    po.Register("chain", &chain, "input chain gmm model");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    is_chain_gmm = chain;

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

    // 3.
    // print head ~o
    std::cout << "~o\n<STREAMINFO> 1 " << am_gmm.Dim() << "\n<VECSIZE> " << am_gmm.Dim() << "<NULLD><USER><DIAGC>\n";
    // print state: ~s
    for (int s = 0; s < trans_model.NumPdfs(); ++s) {
      std::cout << "~s \"PDFID_" << s << "\"" << std::endl;
      std::cout << "<NUMMIXES> " << am_gmm.NumGaussInPdf(s) << std::endl;
      Vector<BaseFloat> weight(am_gmm.GetPdf(s).weights());
      Matrix<BaseFloat> means(am_gmm.NumGaussInPdf(s), am_gmm.Dim());
      am_gmm.GetPdf(s).GetMeans(&means);
      Matrix<BaseFloat> vars(am_gmm.NumGaussInPdf(s), am_gmm.Dim());
      am_gmm.GetPdf(s).GetVars(&vars);
//      Vector<BaseFloat> gconst(am_gmm.GetPdf(s).gconsts());
      for (int m = 0; m < am_gmm.NumGaussInPdf(s); ++m) {
        std::cout << "<MIXTURE> " << m+1 << " " << weight(m) << std::endl;
        std::cout << "<MEAN> " << am_gmm.Dim() << std::endl;
        for (int i = 0; i < am_gmm.Dim(); ++i)
          std::cout << " " << means(m, i);
        std::cout << "\n<VARIANCE> " << am_gmm.Dim() << std::endl;
        for (int i = 0; i < am_gmm.Dim(); ++i)
          std::cout << " " << vars(m, i);
        std::cout << "\n<GCONST> " << M_LOG_2PI * am_gmm.Dim() + vars.Row(m).SumLog() << std::endl;
      }
    }


    // for tri+ trees, read the tree stats;  this gives us basically all
    // phones-in-context that may be linked to an individual model
    // (in practice, many of them will be shared, but we plot them anyways)

    // build-tree-questions.h:typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType
    // typedef std::vector<std::pair<EventKeyType,EventValueType> > EventType
    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(acc_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }

    // print hmm: ~h
    // read the tree, get all the leaves
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);
    // now, for each tree stats element, query the tree to get the pdf-id
    for (size_t i = 0; i < stats.size(); ++i) {
      if (stats[i].first[0].second == 0) // only iter first state of seen triphones
        std::cout << EventTypeToString(stats[i].first, phones_symtab, trans_model, ctx_dep);
    }
    // print ~h proto of unseen phones
    Vector<double> occs;
    if (occ_filename != "") {
      bool binary_in;
      Input ki(occ_filename, &binary_in);
      occs.Read(ki.Stream(), binary_in);
    }
    PrintUnseen(std::cout, trans_model, phones_symtab, &occs, ctx_dep);

    delete phones_symtab;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


