//
// Created by songmeixu (songmeixu@outlook.com) on 2018/9/20.
// Copyright (c) 2018 Xiaomi Inc. All rights reserved.
//

#include <algorithm>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

namespace kaldi {

int CheckWav(const WaveData &wave, int allow_min_threshold=100) {
  const Matrix<BaseFloat> &data = wave.Data();
  long num_min_counts = std::count(data.Data(), data.Data()+data.NumCols(),
                                     std::numeric_limits<short int>::min());
  if (num_min_counts >= allow_min_threshold) {
    return 1;
  }
  return 0;
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "check if wave file with overflow\n"
        "\n"
        "Usage: wav-check [options] <wav-rspecifier>\n"
        "e.g. wav-check scp:wav.scp\n";

    int allow_min_threshold = 1;

    ParseOptions po(usage);

    po.Register("allow-min-threshold", &allow_min_threshold, "how many times min(-32768) value "
                                                             "to be allowed in one wav.");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_in_fn = po.GetArg(1);

    int32 num_done = 0, num_error = 0;

    SequentialTableReader<WaveHolder> wav_reader(wav_in_fn);

    for (; !wav_reader.Done(); wav_reader.Next()) {
      if (CheckWav(wav_reader.Value(), allow_min_threshold)) {
        KALDI_LOG << "find error in: " << wav_reader.Key();
        num_error++;
      }
      num_done++;
    }
    KALDI_LOG << "Checked " << num_done << " wave files, with " << num_error << " errors";
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}