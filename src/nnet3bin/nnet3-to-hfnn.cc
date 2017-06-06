//
// Created by songmeixu on 2017-02-03.
//

#include <fstream>
#include <vector>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-simple-component.h"

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet3;

class BaiduHfnnNet {
 public:
  enum Activation {
    Linear = 0,
    Sigmoid = 1
  };
  int m_nLayer;
  int m_nTotalParamNum;
  vector<int32> m_LayerDim;
  vector< vector<BaseFloat> > weight_;
  vector< vector<BaseFloat> > bias_;
  vector<Activation> m_activation_;

  BaiduHfnnNet() :
      m_nLayer(0),
      m_nTotalParamNum(0) { }

  void Write(std::ostream &os, bool binary = true) const {
    if (!os.good()) {
      KALDI_ERR << "Failed to write vector to stream: stream not good";
    }
    KALDI_ASSERT(m_activation_.size() == weight_.size() && bias_.size() == weight_.size());
    if (binary) {
      uint64 row, col;
      for (int32 l = 0; l < m_nLayer; ++l) {
        // 1. print activation

        // 2. print weight
        // print shape
        row = (uint64) m_LayerDim[l];
        col = (uint64) m_LayerDim[l+1];
        KALDI_LOG << "weight row = " << row << " weight col = " << col;
        os.write(reinterpret_cast<const char*>(&row), sizeof(uint64));
        os.write(reinterpret_cast<const char*>(&col), sizeof(uint64));
        // print weight-param
        KALDI_ASSERT(weight_[l].size() == row * col);
        os.write(reinterpret_cast<const char*>(weight_[l].data()), sizeof(BaseFloat) * weight_[l].size());

        // 3. print bias
        // print shape
        row = 1;
        KALDI_LOG << "bias row = " << row << " bias col = " << col;
        os.write(reinterpret_cast<const char*>(&row), sizeof(uint64));
        os.write(reinterpret_cast<const char*>(&col), sizeof(uint64));
        // print bias-param
        KALDI_ASSERT(bias_[l].size() == col);
        os.write(reinterpret_cast<const char*>(bias_[l].data()), sizeof(BaseFloat) * bias_[l].size());
      }
    } else {
    }
    if (!os.good())
      KALDI_ERR << "Failed to write baidu-net.";
  }

  bool AddToParams(AffineComponent *ac, int layer_idx);
};

bool BaiduHfnnNet::AddToParams(AffineComponent *ac, int layer_idx) {
  assert(layer_idx < m_nLayer);
  const CuMatrix<BaseFloat> weight = ac->LinearParams();
  const CuVector<BaseFloat> &bias = ac->BiasParams();
//  weight.Transpose();

  for (int c = 0; c < weight.NumCols(); ++c) {
    for (int r = 0; r < weight.NumRows(); ++r) {
      weight_[layer_idx].push_back(weight(r, c));
    }
  }
  for (int d = 0; d < ac->OutputDim(); ++d) {
    bias_[layer_idx].push_back(bias(d));
  }

  return true;
}

int main (int argc, const char *argv[]) {
  const char *usage =
      "convert nnet3 mdl to baidu hfnn dnn\n"
          "\n"
          "Usage:  nnet3-to-hfnn [options] <nnet-in> <nnet-out> [priors-out]\n";

  bool binary_write = true;
  kaldi::TransitionModel trans_model;
  AmNnetSimple am_nnet;

  BaiduHfnnNet out_hfnn_net;

  ParseOptions po(usage);
  po.Register("binary", &binary_write, "Read/Write in binary mode");

  po.Read(argc, argv);

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
  out_hfnn_net.m_nLayer = nComponent / 2;
  out_hfnn_net.weight_.resize((unsigned long) out_hfnn_net.m_nLayer);
  out_hfnn_net.bias_.resize((unsigned long) out_hfnn_net.m_nLayer);
  int layer_id = 0;
  for (int i = 0; i < nComponent; ++i) {
    Component *component = am_nnet.GetNnet().GetComponent(i);
    if (component->Type() == "FixedAffineComponent") {
//      kaldi::nnet2::FixedAffineComponent fc = dynamic_cast<kaldi::nnet2::FixedAffineComponent &> (component);
      KALDI_ERR << "currently not support <FixedAffineComponent> in baidu";
      ++layer_id;
    } else if (component->Type() ==  "NaturalGradientAffineComponent") {
      NaturalGradientAffineComponent *ngac =
          dynamic_cast<NaturalGradientAffineComponent *> (component);
      out_hfnn_net.AddToParams(ngac, layer_id);
      if (layer_id == 0) {
        out_hfnn_net.m_LayerDim.push_back(ngac->LinearParams().NumCols());
      }
      out_hfnn_net.m_LayerDim.push_back(ngac->BiasParams().Dim());
      out_hfnn_net.m_nTotalParamNum +=
          ngac->LinearParams().NumRows() * ngac->LinearParams().NumCols()
              + ngac->BiasParams().Dim();
      ++layer_id;
    } else if (component->Type() == "SigmoidComponent") {
      out_hfnn_net.m_activation_.push_back(BaiduHfnnNet::Sigmoid);
    } else if (component->Type() == "LogSoftmaxComponent") {
      out_hfnn_net.m_activation_.push_back(BaiduHfnnNet::Linear);
    }
  }
  Output ko(nnet_wxfilename, binary_write, false);
  out_hfnn_net.Write(ko.Stream(), binary_write);

  // 3. output prior
  if (!prior_wxfilename.empty()) {
    ofstream out(prior_wxfilename.c_str(), ios::out);
    Vector<BaseFloat> priors;
    priors = am_nnet.Priors();
    priors.Write(out, false);
//    for (int i = 0; i < priors.Dim(); ++i) {
//      out.write((char *)&priors(i), sizeof(BaseFloat));
//    }
    out.close();
  }

  return 0;
}

