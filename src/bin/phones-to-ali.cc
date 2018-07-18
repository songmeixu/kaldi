// bin/ali-to-phones.cc

// Copyright 2009-2011  Microsoft Corporation
//           2018       Meixu Song

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

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert phone-sequences (in integer, not text, form) to "
        "model-level alignments, thus the contrary of ali-to-phones\n"
        "Usage: phones-to-ali [options] <model> <phone-transcript-wspecifier> "
        "<alignments-rspecifier>\n"
        "e.g.: \n"
        " phones-to-ali tree 1.mdl ark:1.phones ark:1.ali\n"
        "See also: ali-to-phones\n";
    ParseOptions po(usage);
    BaseFloat frame_shift = 0.01;
    po.Register("frame-shift", &frame_shift,
                "frame shift used to control the times of the ctm output");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        trans_model_rxfilename = po.GetArg(2),
        phone_durs_rspecifier = po.GetArg(3),
        alignment_wspecifier = po.GetArg(4);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);

    SequentialInt32PairVectorReader phone_and_dur_reader(
        phone_durs_rspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int32 N = ctx_dep.ContextWidth();
    int32 P = ctx_dep.CentralPosition();
    const HmmTopology &topo = trans_model.GetTopo();
    int32 num_utts_done = 0;

    for (; !phone_and_dur_reader.Done(); phone_and_dur_reader.Next()) {
      std::string utt = phone_and_dur_reader.Key();
      const std::vector<std::pair<int32,int32> > &phones_durations =
          phone_and_dur_reader.Value();
      KALDI_ASSERT(phones_durations.size() > 0);
      std::vector<int32> phones(phones_durations.size()),
          durations(phones_durations.size());
      for (size_t size = phones_durations.size(), i = 0; i < size; i++) {
        phones[i] = phones_durations[i].first;
        durations[i] = phones_durations[i].second;
      }

      std::vector<int32> alignment;

      for (size_t size = phones.size(), i = 0; i < size; i++) {
        int32 num_states = topo.NumPdfClasses(phones[i]);

        if (durations[i] < 5) {
          durations[i] = 5;
          durations[i+1] = durations[i] + durations[i+1] - 5;
        }
        int32 mean_loop_dur = (durations[i] - num_states) / num_states;
        KALDI_ASSERT(mean_loop_dur >= 0);

        std::vector<int32> ctx_phns(N, 0),
            durs(num_states, mean_loop_dur);
        durs[num_states/2] += (durations[i] - num_states) % num_states;

        if (phones[i] != 1)
          for (int32 offset = 0; offset < N; offset++)
            ctx_phns[offset] = phones[i - N/2 + offset];
        else
          ctx_phns[P] = phones[i];

        for (int32 pdf_class = 0;
             pdf_class < num_states;
             pdf_class++) {
          int32 pdf_id;

          if (!ctx_dep.Compute(ctx_phns, pdf_class, &pdf_id))
            KALDI_ERR << "Cannot find pdf-id for phones: " << phones[i];
          int32 transition_state = trans_model.TupleToTransitionState(
              phones[i], pdf_class, pdf_id, pdf_id);
          int32 transition_id_loop = trans_model.PairToTransitionId(
              transition_state, 0);
          int32 transition_id_jump = trans_model.PairToTransitionId(
              transition_state, 1);
          alignment.push_back(transition_id_jump);
          alignment.insert(alignment.end(), durs[pdf_class], transition_id_loop);
        }
      }
      num_utts_done++;

      if (alignment_writer.IsOpen())
        alignment_writer.Write(utt, alignment);
    }

    KALDI_LOG << "Done " << num_utts_done << " utterances.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


