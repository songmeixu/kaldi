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
#include "matrix/intel_sse.h"

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet2;

typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

using namespace FixedPoint;

class BaiduNet {
 public:
  enum Activation {
    Linear = 0,
    Sigmoid = 1
  };
  int m_nLayer;
  int m_nTotalParamNum;
  vector<int> m_LayerDim;
  vector<BaseFloat> params_;
  bool is_svd;
  vector<int> m_svdDim;
  bool m_is_fixed_;
  int32 m_fixed_bits_;
  vector<float> m_fixed_weight_scales_;
  vector< vector<FPWeight> > m_fixed_weight_;
  vector< vector<BaseFloat> > bias_;
  vector<Activation> m_activation_;

  BaiduNet() :
      m_nLayer(0),
      m_nTotalParamNum(0),
      is_svd(false),
      m_is_fixed_(false),
      m_fixed_bits_(8) {}

  void Write(std::ostream &os, bool binary = true) const {
    if (!os.good()) {
      KALDI_ERR << "Failed to write vector to stream: stream not good";
    }
    KALDI_ASSERT(m_activation_.size() == m_fixed_weight_.size() && bias_.size() == m_fixed_weight_.size());
    if (binary) {
      os.write(reinterpret_cast<const char*>(&m_nLayer), sizeof(int32));
      uint64 row, col;
      for (int l = 0; l < m_nLayer; ++l) {
        // 1. print activation
        os.write(reinterpret_cast<const char*>(&m_activation_[l]), sizeof(int32));

        // 2. print weight
        // print shape
        row = m_LayerDim[l];
        col = m_LayerDim[l+1];
        KALDI_LOG << "weight row = " << row << " weight col = " << col;
        os.write(reinterpret_cast<const char*>(&row), sizeof(uint64));
        os.write(reinterpret_cast<const char*>(&col), sizeof(uint64));
        // print scale
        if (m_is_fixed_) {
          os.write(reinterpret_cast<const char*>(&m_fixed_weight_scales_[l]), sizeof(float));
        }
        // print weight-param
        KALDI_ASSERT(m_fixed_weight_[l].size() == row * col);
        os.write(reinterpret_cast<const char*>(m_fixed_weight_[l].data()), sizeof(FPWeight) * m_fixed_weight_[l].size());

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

  bool AddToParams(AffineComponentFixedPoint &ac, int layer_idx);
};

bool BaiduNet::AddToParams(AffineComponentFixedPoint &ac, int layer_idx) {
  assert(layer_idx < m_nLayer);
  const FixedPoint::Matrix<FPWeight> &weight = ac.FixedWeight();
  const CuVector<BaseFloat> &bias = ac.BiasParams();
//  weight.Transpose();

  if (m_is_fixed_) {
    for (int c = 0; c < weight.NumCols(); ++c) {
      for (int r = 0; r < weight.NumRows(); ++r) {
        m_fixed_weight_[layer_idx].push_back((FPWeight) weight(r, c));
      }
    }
    m_fixed_weight_scales_.push_back(ac.GetWeightScale());
    for (int d = 0; d < ac.OutputDim(); ++d) {
      bias_[layer_idx].push_back((BaseFloat) bias(d));
    }
  } else {
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

  po.Read(argc, argv);

  fixed_write = true;
  out_net.m_is_fixed_ = true;

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
  out_net.m_nLayer = (nComponent - 1) / 2;
  out_net.m_fixed_weight_.resize(out_net.m_nLayer);
  out_net.bias_.resize(out_net.m_nLayer);
  int layer_id = 0;
  for (int i = 0; i < nComponent; ++i) {
    kaldi::nnet2::Component &component = am_nnet.GetNnet().GetComponent(i);
    if (component.Type() == "FixedAffineComponent") {
//      kaldi::nnet2::FixedAffineComponent fc = dynamic_cast<kaldi::nnet2::FixedAffineComponent &> (component);
      KALDI_ERR << "currently not support <FixedAffineComponent> in baidu";
      ++layer_id;
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "AffineComponentPreconditionedOnline") {
      kaldi::nnet2::AffineComponentPreconditionedOnline &acpo = dynamic_cast<kaldi::nnet2::AffineComponentPreconditionedOnline &> (component);
//      AddToParams(out_net, acpo);
      if (layer_id == 0) {
        out_net.m_LayerDim.push_back(acpo.LinearParams().NumCols());
      }
      out_net.m_LayerDim.push_back(acpo.BiasParams().Dim());
      out_net.m_nTotalParamNum += acpo.LinearParams().NumRows() * acpo.LinearParams().NumCols() + acpo.BiasParams().Dim();
      ++layer_id;
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "AffineComponentLRScalePreconditionedOnline") {
      kaldi::nnet2::AffineComponentLRScalePreconditionedOnline &acpo = dynamic_cast<kaldi::nnet2::AffineComponentLRScalePreconditionedOnline &> (component);
      if (layer_id == 0) {
        out_net.m_LayerDim.push_back(acpo.LinearParams().NumCols());
      }
      out_net.is_svd = true;
//      AddToParams(out_net, acpo, false);
      out_net.m_svdDim.push_back(acpo.LinearParams().NumRows());
      out_net.m_nTotalParamNum += acpo.LinearParams().NumRows() * acpo.LinearParams().NumCols();
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "AffineComponentFixedPoint") {
      AffineComponentFixedPoint &acfp = dynamic_cast<AffineComponentFixedPoint &> (component);
      out_net.AddToParams(acfp, layer_id);
      if (layer_id == 0) {
        out_net.m_LayerDim.push_back(acfp.InputDim());
      }
      out_net.m_LayerDim.push_back(acfp.OutputDim());
      out_net.m_nTotalParamNum += acfp.OutputDim() * acfp.InputDim() + acfp.OutputDim();
      ++layer_id;
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "SigmoidComponent") {
      out_net.m_activation_.push_back(BaiduNet::Sigmoid);
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "SoftmaxComponent") {
      out_net.m_activation_.push_back(BaiduNet::Linear);
    }
  }
  Output ko(nnet_wxfilename, binary_write, false);
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

