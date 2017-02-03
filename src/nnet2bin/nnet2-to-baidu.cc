//
// Created by songmeixu on 2017-02-03.
//


#include <fstream>
#include <vector>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute.h"

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet2;

typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

class BaiduNet : AmNnet {
 public:
  int m_nLayer;
  int m_nTotalParamNum;
  vector<int> m_LayerDim;
  vector<BaseFloat> params_;
  bool is_svd;
  vector<int> m_svdDim;
  bool is_fixed;
  int32 fixed_bits;
  vector<float> m_LayerScales;

  BaiduNet() :
      m_nLayer(0),
      m_nTotalParamNum(0) ,
      is_svd(false),
      is_fixed(false),
      fixed_bits(32) {}

  void Write(std::ostream &os, bool binary = true) const {
    if (!os.good()) {
      KALDI_ERR << "Failed to write vector to stream: stream not good";
    }
    if (binary) {
      os.write(reinterpret_cast<const char*>(&m_nLayer), sizeof(int));
      for (int l = 0; l < m_nLayer; ++l) {
        // print shape
        os.write(reinterpret_cast<const char*>(&m_LayerDim[l]), sizeof(int));
        os.write(reinterpret_cast<const char*>(&m_LayerDim[l+1]), sizeof(int));
        // print scale
        if (is_fixed) {
          os.write(reinterpret_cast<const char*>(&m_LayerScales[l]), sizeof(int));
        }
        os.write(reinterpret_cast<const char*>(params_.data()), sizeof(BaseFloat) * params_.size());
      }
    } else {
    }
    if (!os.good())
      KALDI_ERR << "Failed to write baidu-net.";
  }
};

bool AddToParams(BaiduNet &nnet, AffineComponent &ac, bool bias = true) {
  CuMatrix<BaseFloat> weight = ac.LinearParams();
//  weight.Transpose();

  if (!nnet.is_fixed) {
    for (int r = 0; r < weight.NumRows(); ++r) {
      for (int c = 0; c < weight.NumCols(); ++c) {
        nnet.params_.push_back((BaseFloat) weight(r, c));
      }
    }
    if (bias) {
      for (int d = 0; d < ac.BiasParams().Dim(); ++d) {
        nnet.params_.push_back(ac.BiasParams()(d));
      }
    }
  } else {
    weight.
  }

  return true;
}

int main (int argc, const char *argv[]) {
  const char *usage =
      "convert nnet2 mdl to qihoo mars dnn\n"
          "\n"
          "Usage:  nnet2-to-mars [options] <nnet-in> <nnet-out> [priors-out]\n";

  bool binary_write = true;
  bool fixed_write = false;
  int32 fixed_bits = 0;
  kaldi::TransitionModel trans_model;
  kaldi::nnet2::AmNnet am_nnet;

  BaiduNet out_net;

  ParseOptions po(usage);
  po.Register("binary", &binary_write, "Read/Write in binary mode");
  po.Register("fixed-bits", &fixed_bits, "Fixed-point quantization in this bits");

  po.Read(argc, argv);

  if (fixed_bits > 0) {
    fixed_write = true;
    out_net.is_fixed = true;
    out_net.fixed_bits = fixed_bits;
    KALDI_LOG << "Using fixed-bits: " << fixed_bits << " in Fixed-point quantization";
  }

  if (po.NumArgs() < 2 || po.NumArgs() > 3) {
    po.PrintUsage();
    exit(1);
  }

  std::string nnet_rxfilename = po.GetArg(1),
      nnet_wxfilename = po.GetArg(2),
      prior_wxfilename = po.GetOptArg(3);

  // 1. read nnet2
  {
    bool binary;
    Input ki(nnet_rxfilename, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
  }

  // 2. transfer to baidu
  int nComponent = am_nnet.GetNnet().NumComponents();
  int layer_id = 0;
  for (int i = 0; i < nComponent; ++i) {
    kaldi::nnet2::Component &component = am_nnet.GetNnet().GetComponent(i);
    if (component.Type() == "FixedAffineComponent") {
//      kaldi::nnet2::FixedAffineComponent fc = dynamic_cast<kaldi::nnet2::FixedAffineComponent &> (component);
      KALDI_ERR << "currently not support <FixedAffineComponent> in baidu";
      ++layer_id;
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "AffineComponentPreconditionedOnline") {
      kaldi::nnet2::AffineComponentPreconditionedOnline &acpo = dynamic_cast<kaldi::nnet2::AffineComponentPreconditionedOnline &> (component);
      AddToParams(out_net, acpo);
      if (out_net.m_nLayer == 0) {
        out_net.m_LayerDim.push_back(acpo.LinearParams().NumCols());
        ++out_net.m_nLayer;
      }
      out_net.m_LayerDim.push_back(acpo.BiasParams().Dim());
      ++out_net.m_nLayer;
      out_net.m_nTotalParamNum += acpo.LinearParams().NumRows() * acpo.LinearParams().NumCols() + acpo.BiasParams().Dim();
      ++layer_id;
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "AffineComponentLRScalePreconditionedOnline") {
      kaldi::nnet2::AffineComponentLRScalePreconditionedOnline &acpo = dynamic_cast<kaldi::nnet2::AffineComponentLRScalePreconditionedOnline &> (component);
      if (out_net.m_nLayer == 0) {
        out_net.m_LayerDim.push_back(acpo.LinearParams().NumCols());
        ++out_net.m_nLayer;
      }
      out_net.is_svd = true;
      AddToParams(out_net, acpo, false);
      out_net.m_svdDim.push_back(acpo.LinearParams().NumRows());
      out_net.m_nTotalParamNum += acpo.LinearParams().NumRows() * acpo.LinearParams().NumCols();
    }
  }
  Output ko(nnet_wxfilename, binary_write);
  out_net.Write(ko.Stream(), binary_write);

  // 3. output prior
  if (!prior_wxfilename.empty()) {
    ofstream out(prior_wxfilename.c_str(), ios::out | ios::binary);
    Vector<BaseFloat> priors;
    priors = am_nnet.Priors();
    for (int i = 0; i < priors.Dim(); ++i) {
      out.write((char *)&priors(i), sizeof(BaseFloat));
    }
    out.close();
  }

  return 0;
}

