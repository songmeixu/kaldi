// Copyright 2009-2012  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)
//           2018       Meixu Song
//
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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"

#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-utils.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Get alignments of speech.\n"
        "\n"
        "Usage:  speech-aligner [options...] <wav-rspecifier> <transcriptions-rspecifier> <alignments-wspecifier>\n"
        "e.g.: \n"
        " speech-aligner wav.scp 'ark:sym2int.pl -f 2- words.txt text|' ark:out.ali";

    ParseOptions po(usage);
    // feats
    MfccOptions mfcc_opts;
    bool subtract_mean = false;
    BaseFloat vtln_warp = 1.0;
    std::string vtln_map_rspecifier;
    std::string utt2spk_rspecifier;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;
    mfcc_opts.Register(&po);

    // graph
    std::string tree_rxfilename;
    std::string model_rxfilename;
    std::string lex_rxfilename;
    TrainingGraphCompilerOptions gopts;
    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    std::string disambig_rxfilename;
    gopts.Register(&po);

    // align
    std::string scores_wspecifier;
    AlignConfig align_config;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    std::string per_frame_acwt_wspecifier;
    align_config.Register(&po);

    // Register the options
    // feats
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each "
                                                 "feature file [CMS]; not recommended to do it this way. ");
    po.Register("vtln-warp", &vtln_warp, "Vtln warp factor (only applicable "
                                         "if vtln-map not specified)");
    po.Register("vtln-map", &vtln_map_rspecifier, "Map from utterance or "
                                                  "speaker-id to vtln warp factor (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier, "Utterance to speaker-id map "
                                                "rspecifier (if doing VTLN and you have warps per speaker)");
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
                                     "0 -> left, 1 -> right)");
    po.Register("min-duration", &min_duration, "Minimum duration of segments "
                                               "to process (in seconds).");
    // graph
    po.Register("tree-rxfilename", &tree_rxfilename, "tree");
    po.Register("model-rxfilename", &model_rxfilename, "model");
    po.Register("lex-rxfilename", &lex_rxfilename, "lexicon");
    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                                                            "list of disambiguation symbols in phone symbol table");

    // align
    po.Register("scores-wspecifier", &scores_wspecifier, "score");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("write-per-frame-acoustic-loglikes", &per_frame_acwt_wspecifier,
                "Wspecifier for table of vectors containing the acoustic log-likelihoods "
                "per frame for each utterance. E.g. ark:foo/per_frame_logprobs.1.ark");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    // feats
    std::string wav_rspecifier = po.GetArg(1);
    Mfcc mfcc(mfcc_opts);
    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
    TableWriter<HtkMatrixHolder> htk_writer;
    if (utt2spk_rspecifier != "")
      KALDI_ASSERT(vtln_map_rspecifier != "" && "the utt2spk option is only "
                                                "needed if the vtln-map option is used.");
    RandomAccessBaseFloatReaderMapped vtln_map_reader(vtln_map_rspecifier,
                                                      utt2spk_rspecifier);

    // graph
    std::string transcript_rspecifier = po.GetArg(2);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    // need VectorFst because we will change it by adding subseq symbol.
    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;

    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

    lex_fst = NULL;  // we gave ownership to gc.

    SequentialInt32VectorReader transcript_reader(transcript_rspecifier);

    // align
    std::string alignment_wspecifier = po.GetArg(3);
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Int32VectorWriter alignment_writer(alignment_wspecifier);
    BaseFloatWriter scores_writer(scores_wspecifier);
    BaseFloatVectorWriter per_frame_acwt_writer(per_frame_acwt_wspecifier);

    int32 num_utts = 0, num_success = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !reader.Done() || !transcript_reader.Done(); reader.Next(), transcript_reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      KALDI_ASSERT(utt == transcript_reader.Key() && "wav and text key is not equal");

      // feats, features
      const WaveData &wave_data = reader.Value();
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        num_err++;
        continue;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            num_err++;
            continue;
          }
        }
      }
      BaseFloat vtln_warp_local;  // Work out VTLN warp factor.
      if (vtln_map_rspecifier != "") {
        if (!vtln_map_reader.HasKey(utt)) {
          KALDI_WARN << "No vtln-map entry for utterance-id (or speaker-id) "
                     << utt;
          num_err++;
          continue;
        }
        vtln_warp_local = vtln_map_reader.Value(utt);
      } else {
        vtln_warp_local = vtln_warp;
      }
      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      Matrix<BaseFloat> features;
      try {
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), vtln_warp_local, &features);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
                   << utt;
        num_err++;
        continue;
      }
      if (subtract_mean) {
        Vector<BaseFloat> mean(features.NumCols());
        mean.AddRowSumMat(1.0, features);
        mean.Scale(1.0 / features.NumRows());
        for (int32 i = 0; i < features.NumRows(); i++)
          features.Row(i).AddVec(-1.0, mean);
      }

      //graph, decode_fst
      VectorFst<StdArc> decode_fst;
      const std::vector<int32> &transcript = transcript_reader.Value();
      if (!gc.CompileGraphFromText(transcript, &decode_fst)) {
        decode_fst.DeleteStates();  // Just make it empty.
      }
      if (decode_fst.Start() != fst::kNoStateId) {
        num_success++;
      } else {
        KALDI_WARN << "Empty decoding graph for utterance "
                   << utt;
        num_err++;
        continue;
      }
      KALDI_LOG << "compile-train-graphs: succeeded for " << num_success
                << " graphs, failed for " << num_err;

      // align,
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_err++;
        continue;
      }
      {  // Add transition-probs to the FST.
        std::vector<int32> disambig_syms;  // empty.
        AddTransitionProbs(trans_model, disambig_syms,
                           transition_scale, self_loop_scale,
                           &decode_fst);
      }
      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);
      KALDI_LOG << utt;
      AlignUtteranceWrapper(align_config, utt,
                            acoustic_scale, &decode_fst, &gmm_decodable,
                            &alignment_writer, &scores_writer,
                            &num_success, &num_err, &num_retry,
                            &tot_like, &frame_count, &per_frame_acwt_writer);

      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }

    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

