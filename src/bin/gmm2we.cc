//
// Created by songmeixu on 2019-01-29.
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

static bool is_chain_gmm = false;

std::string operator*(std::string const &s, size_t n) {
  std::string r;  // empty string
  r.reserve(n * s.size());
  for (size_t i = 0; i < n; i++)
    r += s;
  return r;
}

// Generate a string representation of the given EventType;  the symtable is
// optional, so is the request for positional symbols (tri-phones: 0-left,
// 1-center, 2-right.
static int gmm2htk(std::ostream &hmmdef_stream,
                   std::ostream &statemap_stream,
                   EventType &e,
                   const fst::SymbolTable *phones_symtab,
                   const TransitionModel &trans_model,
                   const ContextDependency &ctx_dep) {
  // make sure it's sorted so that the kPdfClass is the first element!
  std::sort(e.begin(), e.end());

  // first plot the pdf-class

  std::vector<int32> phone_window;
  std::stringstream context_phns;
  hmmdef_stream << "~h \"";
  for (size_t i = 1; i < e.size(); ++i) {
    if (i == 2)
      context_phns << "-";
    else if (i == 3)
      context_phns << "+";

    phone_window.push_back(e[i].second);
    if (e[i].second == 0)
      return 0; // skip <eps>
    std::string phn = phones_symtab->Find(static_cast<kaldi::int64>(e[i].second));
    if (phn.empty()) {
      // in case we can't resolve the symbol, plot the ID
      KALDI_WARN << "No phone found for ID " << e[i].second;
      context_phns << e[i].second;
    } else
      context_phns << phn;
  }
  hmmdef_stream << context_phns.str() << "\"" << std::endl;
  statemap_stream << context_phns.str() << " ";

  int32 P = ctx_dep.CentralPosition();
  int32 central_phone = phone_window[P];
  const HmmTopology &topo = trans_model.GetTopo();
  const HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(central_phone);
  int num_states = entry.size() - 1;
  int32 num_pdf_classes = topo.NumPdfClasses(central_phone);
  std::vector<int32> pdf_ids(num_pdf_classes);

  for (int32 pdf_class = 0; pdf_class < num_pdf_classes; pdf_class++) {
    if (!ctx_dep.Compute(phone_window, pdf_class, &(pdf_ids[pdf_class]))) {
      std::ostringstream tmp;
      WriteIntegerVector(tmp, false, phone_window);
      KALDI_ERR << "tree did not succeed in converting phone window "
                << tmp.str();
    }
  }

  hmmdef_stream << "<BEGINHMM>" << std::endl;
  hmmdef_stream << "<NUMSTATES> " << num_states + 2 << std::endl;
  statemap_stream << num_states;

  for (int32 hmm_state = 0;
       hmm_state < static_cast<int32>(entry.size());
       hmm_state++) {
    int32 forward_pdf_class = entry[hmm_state].forward_pdf_class, forward_pdf;
    int32 self_loop_pdf_class = entry[hmm_state].self_loop_pdf_class, self_loop_pdf;
    if (forward_pdf_class == kNoPdf) {  // nonemitting state.
      forward_pdf = kNoPdf;
      self_loop_pdf = kNoPdf;
    } else {
      KALDI_ASSERT(forward_pdf_class < static_cast<int32>(pdf_ids.size()));
      KALDI_ASSERT(self_loop_pdf_class < static_cast<int32>(pdf_ids.size()));
      forward_pdf = pdf_ids[forward_pdf_class];
      self_loop_pdf = pdf_ids[self_loop_pdf_class];

      hmmdef_stream << "<STATE> " << hmm_state + 2 << std::endl;
      hmmdef_stream << "~s \"PDFID_" << forward_pdf << "\"" << std::endl;
      statemap_stream << " " << forward_pdf;
      if (is_chain_gmm)
        statemap_stream << " " << self_loop_pdf;

      // print transion prob
      hmmdef_stream << "<TRANSP> " << num_states + 2 << std::endl;
      hmmdef_stream << " 0 1" << std::string(" 0") * num_states << std::endl;
      int32 trans_state;
      int32 trans_id;
      BaseFloat loop_prob;
      BaseFloat out_prob;

      trans_state = trans_model.TupleToTransitionState(central_phone, hmm_state, forward_pdf, self_loop_pdf);
      trans_id = trans_model.PairToTransitionId(trans_state, 0);
      loop_prob = trans_model.GetTransitionProb(trans_id);
      trans_id = trans_model.PairToTransitionId(trans_state, 1);
      out_prob = trans_model.GetTransitionProb(trans_id);
      if (is_chain_gmm && hmm_state == 0)
        hmmdef_stream << " 0 0 " << loop_prob << " " << out_prob << std::endl;
      else
        hmmdef_stream << " 0 " << std::string("0 ") * hmm_state << loop_prob << " " << out_prob
                      << std::string(" 0") * (num_states - hmm_state - 1) << std::endl;

    }
  }

  hmmdef_stream << std::string(" 0") * (num_states + 2) << std::endl;
  hmmdef_stream << "<ENDHMM>" << std::endl;
  statemap_stream << " " << central_phone << std::endl;

  return 0;
}

void PrintUnseen(std::ostream &hmmdef_stream,
                 std::ostream &statemap_stream,
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
    std::vector<int32> phone_window(3, 1);
    phone_window[1] = *it;

    int32 P = ctx_dep.CentralPosition();
    int32 central_phone = phone_window[P];
    const HmmTopology &topo = trans_model.GetTopo();
    const HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(central_phone);
    int num_states = entry.size() - 1;
    int32 num_pdf_classes = topo.NumPdfClasses(central_phone);
    std::vector<int32> pdf_ids(num_pdf_classes);

    for (int32 pdf_class = 0; pdf_class < num_pdf_classes; pdf_class++) {
      if (!ctx_dep.Compute(phone_window, pdf_class, &(pdf_ids[pdf_class]))) {
        std::ostringstream tmp;
        WriteIntegerVector(tmp, false, phone_window);
        KALDI_ERR << "tree did not succeed in converting phone window "
                  << tmp.str();
      }
    }

    hmmdef_stream << "~h \"sil-" << phn << "+sil\"" << std::endl;
    hmmdef_stream << "<BEGINHMM>" << std::endl;
    hmmdef_stream << "<NUMSTATES> " << num_states + 2 << std::endl;
    statemap_stream << "sil-" << phn << "+sil" << " " << num_states;

    hmmdef_stream << "<BEGINHMM>" << std::endl;
    hmmdef_stream << "<NUMSTATES> " << num_states + 2 << std::endl;
    statemap_stream << num_states;

    for (int32 hmm_state = 0;
         hmm_state < static_cast<int32>(entry.size());
         hmm_state++) {
      int32 forward_pdf_class = entry[hmm_state].forward_pdf_class, forward_pdf;
      int32 self_loop_pdf_class = entry[hmm_state].self_loop_pdf_class, self_loop_pdf;
      if (forward_pdf_class == kNoPdf) {  // nonemitting state.
        forward_pdf = kNoPdf;
        self_loop_pdf = kNoPdf;
      } else {
        KALDI_ASSERT(forward_pdf_class < static_cast<int32>(pdf_ids.size()));
        KALDI_ASSERT(self_loop_pdf_class < static_cast<int32>(pdf_ids.size()));
        forward_pdf = pdf_ids[forward_pdf_class];
        self_loop_pdf = pdf_ids[self_loop_pdf_class];

        hmmdef_stream << "<STATE> " << hmm_state + 2 << std::endl;
        hmmdef_stream << "~s \"PDFID_" << forward_pdf << "\"" << std::endl;
        statemap_stream << " " << forward_pdf;
        if (is_chain_gmm)
          statemap_stream << " " << self_loop_pdf;

        // print transion prob
        hmmdef_stream << "<TRANSP> " << num_states + 2 << std::endl;
        hmmdef_stream << " 0 1" << std::string(" 0") * num_states << std::endl;
        int32 trans_state;
        int32 trans_id;
        BaseFloat loop_prob;
        BaseFloat out_prob;

        trans_state = trans_model.TupleToTransitionState(central_phone, hmm_state, forward_pdf, self_loop_pdf);
        trans_id = trans_model.PairToTransitionId(trans_state, 0);
        loop_prob = trans_model.GetTransitionProb(trans_id);
        trans_id = trans_model.PairToTransitionId(trans_state, 1);
        out_prob = trans_model.GetTransitionProb(trans_id);
        if (is_chain_gmm && hmm_state == 0)
          hmmdef_stream << " 0 0 " << loop_prob << " " << out_prob << std::endl;
        else
          hmmdef_stream << " 0 " << std::string("0 ") * hmm_state << loop_prob << " " << out_prob
                        << std::string(" 0") * (num_states - hmm_state - 1) << std::endl;
      }
    }

    hmmdef_stream << std::string(" 0") * (num_states + 2) << std::endl;
    hmmdef_stream << "<ENDHMM>" << std::endl;
    statemap_stream << " " << phone_window[1] << std::endl;
  }
}

int main(int argc, const char *argv[]) {
  try {
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "convert kaldi hmm.mdl to state map, that used for building decode graph\n"
        "Usage:  gmm2we [option] <phones-symbol-table> <in-kaldi-mdl-file> <treeacc> <*.occs> <tree>\n"
        "e.g.: \n"
        "gmm2we phones.txt final.mdl treeacc occs tree hmmdefs state.map\n";

    bool chain = false;
    ParseOptions po(usage);
    po.Register("chain", &chain, "input chain gmm model");

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    is_chain_gmm = chain;

    std::string phones_symtab_filename = po.GetArg(1),
        model_in_filename = po.GetArg(2),
        acc_filename = po.GetOptArg(3),
        occ_filename = po.GetOptArg(4),
        tree_filename = po.GetOptArg(5),
        hmmdef_filename = po.GetOptArg(6),
        statemap_filename = po.GetOptArg(7);

    std::ofstream hmmdef_file(hmmdef_filename);
    std::ofstream statemap_file(statemap_filename);

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

    // 3. convert kaldi
    /// 3.1 print head ~o
    hmmdef_file << "~o\n<STREAMINFO> 1 " << am_gmm.Dim() << "\n<VECSIZE> " << am_gmm.Dim() << "<NULLD><USER><DIAGC>\n";

    /// 3.2 print state: ~s
    for (int s = 0; s < trans_model.NumPdfs(); ++s) {
      hmmdef_file << "~s \"PDFID_" << s << "\"" << std::endl;
      if (am_gmm.NumGaussInPdf(s) > 1)
        hmmdef_file << "<NUMMIXES> " << am_gmm.NumGaussInPdf(s) << std::endl;
      Vector<BaseFloat> weight(am_gmm.GetPdf(s).weights());
      Matrix<BaseFloat> means(am_gmm.NumGaussInPdf(s), am_gmm.Dim());
      am_gmm.GetPdf(s).GetMeans(&means);
      Matrix<BaseFloat> vars(am_gmm.NumGaussInPdf(s), am_gmm.Dim());
      am_gmm.GetPdf(s).GetVars(&vars);
      for (int m = 0; m < am_gmm.NumGaussInPdf(s); ++m) {
        if (am_gmm.NumGaussInPdf(s) > 1)
          hmmdef_file << "<MIXTURE> " << m + 1 << " " << weight(m) << std::endl;
        hmmdef_file << "<MEAN> " << am_gmm.Dim() << std::endl;
        for (int i = 0; i < am_gmm.Dim(); ++i)
          hmmdef_file << " " << means(m, i);
        hmmdef_file << "\n<VARIANCE> " << am_gmm.Dim() << std::endl;
        for (int i = 0; i < am_gmm.Dim(); ++i)
          hmmdef_file << " " << vars(m, i);
        hmmdef_file << "\n<GCONST> " << M_LOG_2PI * am_gmm.Dim() + vars.Row(m).SumLog() << std::endl;
      }
    }

    /// 3.3 print ~h proto for seen phones
    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(acc_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
    //// read the tree, get all the leaves
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);
    //// now, for each tree stats element, query the tree to get the pdf-id
    for (size_t i = 0; i < stats.size(); ++i) {
      if (stats[i].first[0].second == 0) // only iter first state of seen phone_windows
        gmm2htk(hmmdef_file, statemap_file, stats[i].first, phones_symtab, trans_model, ctx_dep);
    }

    /// 3.4 print ~h proto for unseen phones
    Vector<double> occs;
    if (occ_filename != "") {
      bool binary_in;
      Input ki(occ_filename, &binary_in);
      occs.Read(ki.Stream(), binary_in);
    }
    PrintUnseen(hmmdef_file, statemap_file, trans_model, phones_symtab, &occs, ctx_dep);

    delete phones_symtab;
    hmmdef_file.close();
    statemap_file.close();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
