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

#include <locale>
#include <utility>
#include <codecvt>
#include <iomanip>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-mfcc.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"
#include "transform/cmvn.h"
#include "feat/feature-functions.h"

#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"

#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-utils.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

namespace kaldi {

using std::vector;
using std::string;

// returns true if successfully appended.
bool AppendFeats(const std::vector<Matrix<BaseFloat> > &in,
                 const std::string &utt,
                 int32 tolerance,
                 Matrix<BaseFloat> *out) {
  // Check the lengths
  int32 min_len = in[0].NumRows(),
      max_len = in[0].NumRows(),
      tot_dim = in[0].NumCols();
  for (int32 i = 1; i < in.size(); i++) {
    int32 len = in[i].NumRows(), dim = in[i].NumCols();
    tot_dim += dim;
    if(len < min_len) min_len = len;
    if(len > max_len) max_len = len;
  }
  if (max_len - min_len > tolerance || min_len == 0) {
    KALDI_WARN << "Length mismatch " << max_len << " vs. " << min_len
               << (utt.empty() ? "" : " for utt ") << utt
               << " exceeds tolerance " << tolerance;
    out->Resize(0, 0);
    return false;
  }
  if (max_len - min_len > 0) {
    KALDI_VLOG(2) << "Length mismatch " << max_len << " vs. " << min_len
                  << (utt.empty() ? "" : " for utt ") << utt
                  << " within tolerance " << tolerance;
  }
  out->Resize(min_len, tot_dim);
  int32 dim_offset = 0;
  for (const auto i : in) {
    int32 this_dim = i.NumCols();
    out->Range(0, min_len, dim_offset, this_dim).CopyFromMat(
        i.Range(0, min_len, 0, this_dim));
    dim_offset += this_dim;
  }
  return true;
}


std::wstring s2ws(const std::string& str) {
  using convert_typeX = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_typeX, wchar_t> converterX;

  return converterX.from_bytes(str);
}

std::string ws2s(const std::wstring& wstr) {
  using convert_typeX = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_typeX, wchar_t> converterX;

  return converterX.to_bytes(wstr);
}

bool SegWordFMM(fst::SymbolTable *word_syms, const string &sentence,
    vector<string> &words, vector<int32> &word_ids) {
  std::wstring sent = s2ws(sentence);
  int maxLength = 10, index = 0, length = sent.size();
  while (index < length) {
    int wordLen = length - index + 1 >= maxLength ? maxLength : length - index + 1;
    while (wordLen >= 1) {
      std::wstring cur = sent.substr(index, wordLen);
      string curWord = ws2s(cur);
      int32 word_id = word_syms->Find(curWord);
      if (word_id != -1 || 1 == wordLen) {
        words.push_back(curWord);
        word_ids.push_back(word_id);
        index += wordLen;
        break;
      }
      wordLen--;
    }
  }
  return true;
}

}

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
    PitchExtractionOptions pitch_opts;
    ProcessPitchOptions process_opts;
    process_opts.Register(&po);
    int32 length_tolerance = 0;
    bool norm_vars = false;
    bool norm_means = true;
    DeltaFeaturesOptions delta_opts;

    // graph
    std::string tree_rxfilename;
    std::string model_rxfilename;
    std::string lex_rxfilename;
    std::string disambig_rxfilename;
    std::string word_syms_filename;
    TrainingGraphCompilerOptions gopts;
    gopts.Register(&po);

    // align
    AlignConfig align_config;
    BaseFloat acoustic_scale = 0.1;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 0.1;
    align_config.Register(&po);
    bool per_frame = false;
    bool write_lengths = false;
    bool ctm_output = false;
    bool custom_output = true;
    BaseFloat frame_shift = 0.005;
    std::string phone_syms_filename;


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
    po.Register("length-tolerance", &length_tolerance,
                "If length is different, trim as shortest up to a frame "
                " difference of length-tolerance, otherwise exclude segment.");
    po.Register("norm-vars", &norm_vars, "If true, normalize variances.");
    po.Register("norm-means", &norm_means, "You can set this to false to turn off mean "
                                           "normalization.  Note, the same can be achieved by using 'fake' CMVN stats; "
                                           "see the --fake option to compute_cmvn_stats.sh");

    // graph
    po.Register("tree-rxfilename", &tree_rxfilename, "tree");
    po.Register("model-rxfilename", &model_rxfilename, "model");
    po.Register("lex-rxfilename", &lex_rxfilename, "lexicon");
    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                                                            "list of disambiguation symbols in phone symbol table");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words");

    // align
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("ctm-output", &ctm_output,
                "If true, output the alignments in ctm format "
                "(the confidences will be set to 1)");
    po.Register("per-frame", &per_frame,
                "If true, write out the frame-level phone alignment "
                "(else phone sequence)");
    po.Register("write-lengths", &write_lengths,
                "If true, write the #frames for each phone (different format)");
    po.Register("phone-symbol-table", &phone_syms_filename,
                "Symbol table for phones");
    po.Register("custom-output", &custom_output,
                "If true, output in the custom format");


    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    // feats
    std::string wav_rspecifier = po.GetArg(1);
    if (wav_rspecifier.substr(0, 3) != "scp:") {
      wav_rspecifier = "scp:" + wav_rspecifier;
    }
    Mfcc mfcc(mfcc_opts);
    pitch_opts.frame_shift_ms = mfcc_opts.frame_opts.frame_shift_ms;
    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
    TableWriter<HtkMatrixHolder> htk_writer;
    if (!utt2spk_rspecifier.empty())
      KALDI_ASSERT(!vtln_map_rspecifier.empty() && "the utt2spk option is only "
                                                "needed if the vtln-map option is used.");
    RandomAccessBaseFloatReaderMapped vtln_map_reader(vtln_map_rspecifier,
                                                      utt2spk_rspecifier);
    if (norm_vars && !norm_means)
      KALDI_ERR << "You cannot normalize the variance but not the mean.";

    // graph
    std::string trans_file = po.GetArg(2);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    // need VectorFst because we will change it by adding subseq symbol.
    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);

    std::vector<int32> disambig_syms;
    if (!disambig_rxfilename.empty())
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;

    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

    lex_fst = nullptr;  // we gave ownership to gc.

    std::ifstream trans_text(trans_file);

    // align
    std::string alignment_wspecifier = po.GetArg(3);
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    fst::SymbolTable *word_syms = NULL;
    word_syms = fst::SymbolTable::ReadText(word_syms_filename);
    if (!word_syms) {
      KALDI_ERR << "Could not read symbol table from file " << word_syms_filename;
    }

    std::string empty;
    Int32VectorWriter phones_writer(custom_output || ctm_output ? empty :
                                    (write_lengths ? empty : alignment_wspecifier));
    Int32PairVectorWriter pair_writer(custom_output || ctm_output ? empty :
                                      (write_lengths ? alignment_wspecifier : empty));
    std::ofstream output(alignment_wspecifier);

    fst::SymbolTable *phone_syms = NULL;
    phone_syms = fst::SymbolTable::ReadText(phone_syms_filename);
    if (!phone_syms) {
      KALDI_ERR << "Could not read symbol table from file " << phone_syms_filename;
    }

    std::string ctm_wxfilename(ctm_output ? po.GetArg(3) : empty);
    Output ctm_writer(ctm_wxfilename, false);
    if (ctm_output) {
      ctm_writer.Stream() << std::fixed;
      ctm_writer.Stream().precision(frame_shift >= 0.01 ? 2 : 3);
    }

    int32 num_utts = 0, num_success = 0, num_err = 0, num_retry = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    std::string line;

    for (; !wav_reader.Done(); wav_reader.Next()) {
      num_utts++;
      std::string utt = wav_reader.Key();
      KALDI_LOG << utt;

      std::getline(trans_text, line);
      KALDI_ASSERT(!line.empty() && "key of text files is not equal that of wav files");
      std::vector<std::string> items;
      std::istringstream iss(line);
      for(std::string s; iss >> s; )
        items.push_back(s);
      KALDI_ASSERT(items.size() == 2 && "transcript is not in \"key non-blank-characters\" format");
      KALDI_ASSERT(utt == items[0] && "wav and text key is not equal");
      std::string sentence = items[1];
      std::vector<std::string> words;
      std::vector<int32> word_ids;
      SegWordFMM(word_syms, sentence, words, word_ids);

      // feats
      const WaveData &wave_data = wav_reader.Value();
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
      if (!vtln_map_rspecifier.empty()) {
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
      Matrix<BaseFloat> mfcc_feat;
      /// mfcc
      try {
        mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), vtln_warp_local, &mfcc_feat);
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
                   << utt;
        num_err++;
        continue;
      }
      if (subtract_mean) {
        Vector<BaseFloat> mean(mfcc_feat.NumCols());
        mean.AddRowSumMat(1.0, mfcc_feat);
        mean.Scale(1.0f / mfcc_feat.NumRows());
        for (int32 i = 0; i < mfcc_feat.NumRows(); i++)
          mfcc_feat.Row(i).AddVec(-1.0f, mean);
      }
      /// pitch
      if (pitch_opts.samp_freq != wave_data.SampFreq())
        KALDI_ERR << "Sample frequency mismatch: you specified "
                  << pitch_opts.samp_freq << " but data has "
                  << wave_data.SampFreq() << " (use --sample-frequency "
                  << "option).  Utterance is " << utt;
      Matrix<BaseFloat> base_feats;
      try {
        Matrix<BaseFloat> pitch;
        ComputeKaldiPitch(pitch_opts, waveform, &pitch);
        Matrix<BaseFloat> processed_pitch(pitch);
        ProcessPitch(process_opts, pitch, &processed_pitch);

        std::vector<Matrix<BaseFloat> > feats(2);
        feats[0] = mfcc_feat;
        feats[1] = processed_pitch;
        Matrix<BaseFloat> output;
        if (!AppendFeats(feats, utt, length_tolerance, &base_feats)) {
          KALDI_WARN << "Failed to combine mfcc and pitch for utterance "
                     << utt;
          num_err++;
          continue; // it will have printed a warning.
        }
      } catch (...) {
        KALDI_WARN << "Failed to compute pitch for utterance "
                   << utt;
        num_err++;
        continue;
      }
      Matrix<double> cmvn_stats;
      Matrix<BaseFloat> features;
      InitCmvnStats(base_feats.NumCols(), &cmvn_stats);
      AccCmvnStats(base_feats, nullptr, &cmvn_stats);
      ApplyCmvn(cmvn_stats, norm_vars, &base_feats);
      ComputeDeltas(delta_opts, base_feats, &features);

      //graph, decode_fst
      VectorFst<StdArc> decode_fst;
      if (!gc.CompileGraphFromText(word_ids, &decode_fst)) {
        decode_fst.DeleteStates();  // Just make it empty.
      }
      if (decode_fst.Start() == fst::kNoStateId) {
        KALDI_WARN << "Empty decoding graph for utterance "
                   << utt;
        num_err++;
        continue;
      }
      KALDI_VLOG(2) << "compile-train-graphs: succeeded for " << num_success
                    << " graphs, failed for " << num_err;

      // align,
      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << utt;
        num_err++;
        continue;
      }
      {  // Add transition-probs to the FST.
        std::vector<int32> disambig_syms_empty;  // empty.
        AddTransitionProbs(trans_model, disambig_syms_empty,
                           transition_scale, self_loop_scale,
                           &decode_fst);
      }
      DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);
      std::vector<int32> alignment;
      Vector<BaseFloat> per_frame_acwt;
      BaseFloat score;
      AlignOneUtteranceWrapper(align_config, utt,
                               acoustic_scale, &decode_fst, &gmm_decodable,
                               alignment, &score,
                               &num_success, &num_err, &num_retry,
                               &tot_like, &frame_count, &per_frame_acwt);

      if (!alignment.empty()) {
        std::vector<std::vector<int32> > split;
        SplitToPhones(trans_model, alignment, &split);

        if (custom_output) {
          float st = 0.0, et = 0.0;
          output << utt << std::endl;
          for (size_t i = 0; i < split.size(); i++) {
            KALDI_ASSERT(!split[i].empty());
            int32 phone_id = trans_model.TransitionIdToPhone(split[i][0]);
            std::string phone = phone_syms->Find(phone_id);
            int32 num_repeats = split[i].size();
            //KALDI_ASSERT(num_repeats!=0);
            st = et;
            et += num_repeats * frame_shift;
            output << std::fixed << std::setprecision(3) << st << " " << et << " " << phone << std::endl;
          }
          output << "." << std::endl;
        } else if (ctm_output) {
          BaseFloat phone_start = 0.0;
          for (size_t i = 0; i < split.size(); i++) {
            KALDI_ASSERT(!split[i].empty());
            int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
            int32 num_repeats = split[i].size();
            ctm_writer.Stream() << utt << " 1 " << phone_start << " "
                                << (frame_shift * num_repeats) << " " << phone << std::endl;
            phone_start += frame_shift * num_repeats;
          }
        } else if (!write_lengths) {
          std::vector<int32> phones;
          for (size_t i = 0; i < split.size(); i++) {
            KALDI_ASSERT(!split[i].empty());
            int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
            int32 num_repeats = split[i].size();
            //KALDI_ASSERT(num_repeats!=0);
            if (per_frame)
              for(int32 j = 0; j < num_repeats; j++)
                phones.push_back(phone);
            else
              phones.push_back(phone);
          }
          phones_writer.Write(utt, phones);
        } else {
          std::vector<std::pair<int32, int32> > pairs;
          for (size_t i = 0; i < split.size(); i++) {
            KALDI_ASSERT(!split[i].empty());
            int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
            int32 num_repeats = split[i].size();
            //KALDI_ASSERT(num_repeats!=0);
            pairs.push_back(std::make_pair(phone, num_repeats));
          }
          pair_writer.Write(utt, pairs);
        }
      }

      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
    }

    trans_text.close();
    output.close();
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

