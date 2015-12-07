//
// Created by songmeixu on 15/11/19.
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

class MarsNet {
 public:
  int m_nLayer;
  int m_nTotalParamNum;
  string head;
  vector<int> m_LayerDim;
  vector<BaseFloat> params_;

  MarsNet() : m_nLayer(0), m_nTotalParamNum(0) ,head("#MUSE_NETWORK VERSION 1.0") {}

  void Write(std::ostream &os, bool binary) const {
    if (!os.good()) {
      KALDI_ERR << "Failed to write vector to stream: stream not good";
    }
    if (binary) {
      os << "#MUSE_NETWORK VERSION 1.0" << endl;
      os << "#NETWORK TOPOLOGY:\t";
      for (int l = 0; l < m_nLayer; ++l) {
        os << m_LayerDim[l] << (l == m_nLayer - 1 ? "\n" : ",");
      }
      os << m_nTotalParamNum << endl;
      os.write(reinterpret_cast<const char*>(&m_nLayer), sizeof(int));
      os.write(reinterpret_cast<const char*>(m_LayerDim.data()), sizeof(int) * m_LayerDim.size());
      os.write(reinterpret_cast<const char*>(params_.data()), sizeof(BaseFloat) * params_.size());
    } else {
    }
    if (!os.good())
      KALDI_ERR << "Failed to write vector to stream";
  }
};

bool AddToParams(MarsNet &mars_net, AffineComponent &ac) {
  CuMatrix<BaseFloat> weight = ac.LinearParams();
//  weight.Transpose();
  for (int r = 0; r < weight.NumRows(); ++r) {
    for (int c = 0; c < weight.NumCols(); ++c) {
      mars_net.params_.push_back((BaseFloat) weight(r,c));
    }
  }
  for (int d = 0; d < ac.BiasParams().Dim(); ++d) {
    mars_net.params_.push_back(ac.BiasParams()(d));
  }
  return true;
}

int main (int argc, const char *argv[]) {
  const char *usage =
      "convert nnet2 mdl to qihoo mars dnn\n"
          "\n"
          "Usage:  nnet2-to-mars [options] <nnet-in> <nnet-out> [priors-out]\n";

  bool binary_write = true;
  kaldi::TransitionModel trans_model;
  kaldi::nnet2::AmNnet am_nnet;

  MarsNet mars_net;

  ParseOptions po(usage);
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

  // 2. transfer to tit
  int nComponent = am_nnet.GetNnet().NumComponents();
  int layer_id = 0;
  for (int i = 0; i < nComponent; ++i) {
    kaldi::nnet2::Component &component = am_nnet.GetNnet().GetComponent(i);
    if (component.Type() == "FixedAffineComponent") {
//      kaldi::nnet2::FixedAffineComponent fc = dynamic_cast<kaldi::nnet2::FixedAffineComponent &> (component);
      KALDI_ERR << "currently not support <FixedAffineComponent> in mars";
      ++layer_id;
    } else if (am_nnet.GetNnet().GetComponent(i).Type() == "AffineComponentPreconditionedOnline") {
      kaldi::nnet2::AffineComponentPreconditionedOnline &acpo = dynamic_cast<kaldi::nnet2::AffineComponentPreconditionedOnline &> (component);
      AddToParams(mars_net, acpo);
      if (mars_net.m_nLayer == 0) {
        mars_net.m_LayerDim.push_back(acpo.LinearParams().NumCols());
        ++mars_net.m_nLayer;
      }
      mars_net.m_LayerDim.push_back(acpo.BiasParams().Dim());
      ++mars_net.m_nLayer;
      mars_net.m_nTotalParamNum += acpo.LinearParams().NumRows() * acpo.LinearParams().NumCols() + acpo.BiasParams().Dim();
      ++layer_id;
    }
  }
  Output ko(nnet_wxfilename, binary_write);
  mars_net.Write(ko.Stream(), binary_write);

  // 3. output prior
  if (!prior_wxfilename.empty()) {
    Output out(prior_wxfilename, binary_write);
//    kaldi::CuVector<kaldi::BaseFloat> priors;
//    priors = am_nnet.Priors();
//    priors.ApplyLog();
//    priors.Scale(1 / kaldi::Log(10.0));
    am_nnet.Priors().Write(out.Stream(), false);
//     out.Stream().write(reinterpret_cast<const char*>(am_nnet.Priors().Data()), sizeof(BaseFloat) * am_nnet.Priors().Dim());
  }

  return 0;
}

