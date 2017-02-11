//
// Created by songmeixu on 2017/2/11.
//

#include <fstream>
#include <vector>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-binary-component.h"

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet3;

typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

class BaiduNet {
 public:
  int32 m_nLayer;
  int32 m_nTotalParamNum;
  string head;
  vector<int32> m_LayerDim;
  vector<BaseFloat> params_;
  bool is_svd;
  vector<int32> m_svdDim;

  BaiduNet() : m_nLayer(0), m_nTotalParamNum(0) ,head("#MUSE_NETWORK VERSION 1.0"), is_svd(false) {}

  void Write(std::ostream &os, bool binary = true) const {
    if (!os.good()) {
      KALDI_ERR << "Failed to write vector to stream: stream not good";
    }
    if (binary) {
      os << "#MUSE_NETWORK VERSION 1.0" << endl;
      os << "#NETWORK TOPOLOGY:\t";
      for (int32 l = 0; l < m_nLayer; ++l) {
        os << m_LayerDim[l] << (l == m_nLayer - 1 ? "\n" : ",");
      }
      os << m_nTotalParamNum << endl;
      os.write(reinterpret_cast<const char*>(&m_nLayer), sizeof(int));
      os.write(reinterpret_cast<const char*>(m_LayerDim.data()), sizeof(int32) * m_LayerDim.size());
      if (is_svd) {
        os.write(reinterpret_cast<const char*>(m_svdDim.data()), sizeof(int32) * m_svdDim.size());
      }
      os.write(reinterpret_cast<const char*>(params_.data()), sizeof(BaseFloat) * params_.size());
    } else {
    }
    if (!os.good())
      KALDI_ERR << "Failed to write vector to stream";
  }
};

bool AddToParams(BaiduNet &baidu_net, AffineComponent &ac, bool bias = true) {
  CuMatrix<BaseFloat> weight = ac.LinearParams();
//  weight.Transpose();
  for (int32 r = 0; r < weight.NumRows(); ++r) {
    for (int32 c = 0; c < weight.NumCols(); ++c) {
      baidu_net.params_.push_back((BaseFloat) weight(r,c));
    }
  }
  if (bias) {
    for (int32 d = 0; d < ac.BiasParams().Dim(); ++d) {
      baidu_net.params_.push_back(ac.BiasParams()(d));
    }
  }
  return true;
}

bool AddToParams(BaiduNet &baidu_net, BinaryAffineComponent &ac, bool bias = false) {
  CuMatrix<BaseFloat> weight = ac.BinaryLinearParams();
//  weight.Transpose();
  for (int32 r = 0; r < weight.NumRows(); ++r) {
    for (int32 c = 0; c < weight.NumCols(); ++c) {
      baidu_net.params_.push_back((BaseFloat) weight(r,c));
    }
  }
  if (bias) {
    for (int32 d = 0; d < ac.BiasParams().Dim(); ++d) {
      baidu_net.params_.push_back(ac.BiasParams()(d));
    }
  }
  return true;
}

bool AddToParams(BaiduNet &baidu_net, BatchNormComponent &bnc) {
  for (int32 d = 0; d < bnc.OutputDim().Dim(); ++d) {
    baidu_net.params_.push_back(ac.A()(d));
  }
  for (int32 d = 0; d < bnc.OutputDim().Dim(); ++d) {
    baidu_net.params_.push_back(ac.B()(d));
  }
  return true;
}

int main (int argc, const char *argv[]) {
  const char *usage =
      "convert nnet3 mdl to baidu compressed dnn\n"
          "\n"
          "Usage:  nnet3-to-baidu [options] <nnet-in> <nnet-out> [priors-out]\n";

  bool binary_write = true;
  TransitionModel trans_model;
  AmNnetSimple am_nnet;

  BaiduNet baidu_net;

  ParseOptions po(usage);
  po.Read(argc, argv);

  if (po.NumArgs() < 2 || po.NumArgs() > 3) {
    po.PrintUsage();
    exit(1);
  }

  std::string nnet_rxfilename = po.GetArg(1),
      nnet_wxfilename = po.GetArg(2),
      prior_wxfilename = po.GetOptArg(3);

  // 1. read nnet
  {
    bool binary;
    Input ki(nnet_rxfilename, &binary);
    trans_model.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
  }

  // 2. transfer to baidu
  int nComponent = am_nnet.GetNnet().NumComponents();
  int layer_id = 0;
  for (int32 i = 0; i < nComponent; ++i) {
    Component &component = am_nnet.GetNnet().GetComponent(i);
    if (component.Type() == "FixedAffineComponent") {
//      kaldi::nnet2::FixedAffineComponent fc = dynamic_cast<kaldi::nnet2::FixedAffineComponent &> (component);
      KALDI_ERR << "currently not support <FixedAffineComponent> in BaiduNet";
      ++layer_id;
    } else if (component.Type() == "AffineComponentPreconditionedOnline") {
      AffineComponentPreconditionedOnline &acpo = dynamic_cast<AffineComponentPreconditionedOnline &> (component);
      AddToParams(baidu_net, acpo);
      if (baidu_net.m_nLayer == 0) {
        baidu_net.m_LayerDim.push_back(acpo.LinearParams().NumCols());
        ++baidu_net.m_nLayer;
      }
      baidu_net.m_LayerDim.push_back(acpo.BiasParams().Dim());
      ++baidu_net.m_nLayer;
      baidu_net.m_nTotalParamNum += acpo.LinearParams().NumRows() * acpo.LinearParams().NumCols() + acpo.BiasParams().Dim();
      ++layer_id;
    } else if (component.Type() == "BinaryAffineComponent") {
      BinaryAffineComponent &bac = dynamic_cast<BinaryAffineComponent &> (component);
      if (baidu_net.m_nLayer == 0) {
        baidu_net.m_LayerDim.push_back(bac.LinearParams().NumCols());
        ++baidu_net.m_nLayer;
      }
      AddToParams(baidu_net, bac, false);
      baidu_net.m_LayerDim.push_back(bac.BiasParams().Dim());
      ++baidu_net.m_nLayer;
      baidu_net.m_nTotalParamNum += bac.LinearParams().NumRows() * bac.LinearParams().NumCols() + bac.BiasParams().Dim();
    } else if (component.Type() == "BatchNormComponent") {
      BatchNormComponent &bnc = dynamic_cast<BatchNormComponent &> (component);
      AddToParams(baidu_net, bnc);
      baidu_net.m_nTotalParamNum += bnc.OutputDim()*2;
    }
  }
  Output ko(nnet_wxfilename, binary_write);
  baidu_net.Write(ko.Stream(), binary_write);

  // 3. output prior
  if (!prior_wxfilename.empty()) {
    ofstream out(prior_wxfilename.c_str(), ios::out | ios::binary);
    Vector<BaseFloat> priors;
    priors = am_nnet.Priors();
    for (int i = 0; i < priors.Dim(); ++i) {
      out.write((char *)&priors(i), sizeof(BaseFloat));
//      cout << priors(i) << endl;
    }
    out.close();
  }

  return 0;
}

