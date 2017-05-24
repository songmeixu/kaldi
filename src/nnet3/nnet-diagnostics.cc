// nnet3/nnet-diagnostics.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-diagnostics.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetComputeProb::NnetComputeProb(const NnetComputeProbOptions &config,
                                 const Nnet &nnet):
    config_(config),
    nnet_(nnet),
    deriv_nnet_(NULL),
    compiler_(nnet, config_.optimize_config, config_.compiler_config),
    num_minibatches_processed_(0) {
  if (config_.compute_deriv) {
    deriv_nnet_ = new Nnet(nnet_);
    ScaleNnet(0.0, deriv_nnet_);
    SetNnetAsGradient(deriv_nnet_); // force simple update
  }

  if (!config_.pdf_classes.empty()) {
    std::cout << "pdf-id classes: " << config_.pdf_classes << std::endl;
    std::vector<int> pdf_ids;
    std::vector<std::string> classes = SplitStrings(config_.pdf_classes, ':');
    for (int i = 0; i < classes.size(); ++i) {
      if (!SplitStringToIntegers(classes[i], ",", false, &pdf_ids)) {
        KALDI_ERR << "Bad --skip-dims option (should be colon-separated list of "
                  << "integers)";
      }
      pdfid_classes_.push_back(pdf_ids);
    }
  }
}

const Nnet &NnetComputeProb::GetDeriv() const {
  if (deriv_nnet_ == NULL)
    KALDI_ERR << "GetDeriv() called when no derivatives were requested.";
  return *deriv_nnet_;
}

NnetComputeProb::~NnetComputeProb() {
  delete deriv_nnet_;  // delete does nothing if pointer is NULL.
}

void NnetComputeProb::Reset() {
  num_minibatches_processed_ = 0;
  objf_info_.clear();
  accuracy_info_.clear();
  if (deriv_nnet_) {
    ScaleNnet(0.0, deriv_nnet_);
    SetNnetAsGradient(deriv_nnet_);
  }
}

void NnetComputeProb::Compute(const NnetExample &eg) {
  bool need_model_derivative = config_.compute_deriv,
      store_component_stats = false;
  ComputationRequest request;
  GetComputationRequest(nnet_, eg, need_model_derivative,
                        store_component_stats,
                        &request);
  const NnetComputation *computation = compiler_.Compile(request);
  NnetComputer computer(config_.compute_config, *computation,
                        nnet_, deriv_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(nnet_, eg.io);
  computer.Run();
  this->ProcessOutputs(eg, &computer);
  if (config_.compute_deriv)
    computer.Run();
}

void NnetComputeProb::ProcessOutputs(const NnetExample &eg,
                                     NnetComputer *computer) {
  std::vector<NnetIo>::const_iterator iter = eg.io.begin(),
      end = eg.io.end();
  for (; iter != end; ++iter) {
    const NnetIo &io = *iter;
    int32 node_index = nnet_.GetNodeIndex(io.name);
    if (node_index < 0)
      KALDI_ERR << "Network has no output named " << io.name;
    ObjectiveType obj_type = nnet_.GetNode(node_index).u.objective_type;
    if (nnet_.IsOutputNode(node_index)) {
      const CuMatrixBase<BaseFloat> &output = computer->GetOutput(io.name);
      if (output.NumCols() != io.features.NumCols()) {
        KALDI_ERR << "Nnet versus example output dimension (num-classes) "
                  << "mismatch for '" << io.name << "': " << output.NumCols()
                  << " (nnet) vs. " << io.features.NumCols() << " (egs)\n";
      }
      {
        BaseFloat tot_weight, tot_objf;
        bool supply_deriv = config_.compute_deriv;
        ComputeObjectiveFunction(io.features, obj_type, io.name,
                                 supply_deriv, computer,
                                 &tot_weight, &tot_objf);
        SimpleObjectiveInfo &totals = objf_info_[io.name];
        totals.tot_weight += tot_weight;
        totals.tot_objective += tot_objf;
      }
      if (obj_type == kLinear && config_.compute_accuracy) {
        BaseFloat tot_weight, tot_accuracy;
        double *class_accuracy = new double[2*(pdfid_classes_.size()+1)]();
        ComputeAccuracy(io.features, output,
                        &tot_weight, &tot_accuracy, pdfid_classes_, class_accuracy);
        SimpleObjectiveInfo &totals = accuracy_info_[io.name];
        totals.tot_weight += tot_weight;
        totals.tot_objective += tot_accuracy;
        if (totals.tot_class_accuracy == NULL)
          totals.tot_class_accuracy = new double[2*(pdfid_classes_.size()+1)]();
        for (int c = 0; c < 2 * (pdfid_classes_.size() + 1); ++c) {
          totals.tot_class_accuracy[c] += class_accuracy[c];
        }
      }
      num_minibatches_processed_++;
    }
  }
}

bool NnetComputeProb::PrintTotalStats() const {
  bool ans = false;
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter, end;
  { // First print regular objectives
    iter = objf_info_.begin();
    end = objf_info_.end();
    for (; iter != end; ++iter) {
      const std::string &name = iter->first;
      int32 node_index = nnet_.GetNodeIndex(name);
      KALDI_ASSERT(node_index >= 0);
      ObjectiveType obj_type = nnet_.GetNode(node_index).u.objective_type;
      const SimpleObjectiveInfo &info = iter->second;
      KALDI_LOG << "Overall "
                << (obj_type == kLinear ? "log-likelihood" : "objective")
                << " for '" << name << "' is "
                << (info.tot_objective / info.tot_weight) << " per frame"
                << ", over " << info.tot_weight << " frames.";
      if (info.tot_weight > 0)
        ans = true;
    }
  }
  { // now print accuracies.
    iter = accuracy_info_.begin();
    end = accuracy_info_.end();
    for (; iter != end; ++iter) {
      const std::string &name = iter->first;
      const SimpleObjectiveInfo &info = iter->second;
      KALDI_LOG << "Overall accuracy for '" << name << "' is "
                << (info.tot_objective / info.tot_weight) << " per frame"
                << ", over " << info.tot_weight << " frames.";
      // don't bother changing ans; the loop over the regular objective should
      // already have set it to true if we got any data.
    }
  }
  { // print pdf-id class accuracies
    if (!pdfid_classes_.empty()) {
      iter = accuracy_info_.begin();
      end = accuracy_info_.end();
      for (; iter != end; ++iter) {
        const std::string &name = iter->first;
        const SimpleObjectiveInfo &info = iter->second;

        for (int c = 0; c < pdfid_classes_.size(); ++c) {
          double accuracy = info.tot_class_accuracy[2 * c + 1] == 0 ? 0 :
                            (info.tot_class_accuracy[2 * c] / info.tot_class_accuracy[2 * c + 1]);
          KALDI_LOG << "class " << c + 1 << " accuracy is "
                    << accuracy
                    << ", total classes num is "
                    << info.tot_class_accuracy[2 * c + 1];
        }
        KALDI_LOG << "total classes accuracy is "
                  << (info.tot_class_accuracy[2 * pdfid_classes_.size()] /
                      info.tot_class_accuracy[2 * pdfid_classes_.size() + 1])
                  << ", total classes num is "
                  << info.tot_class_accuracy[2 * pdfid_classes_.size() + 1];
      }
    }
  }
  return ans;
}

void ComputeAccuracy(const GeneralMatrix &supervision,
                     const CuMatrixBase<BaseFloat> &nnet_output,
                     BaseFloat *tot_weight_out,
                     BaseFloat *tot_accuracy_out,
                     const std::vector< std::vector<int> > &pdfid_classes,
                     double *tot_classes_accuracy) {
  int32 num_rows = nnet_output.NumRows(),
      num_cols = nnet_output.NumCols();
  KALDI_ASSERT(supervision.NumRows() == num_rows &&
               supervision.NumCols() == num_cols);

  CuArray<int32> best_index(num_rows);
  nnet_output.FindRowMaxId(&best_index);
  std::vector<int32> best_index_cpu;
  // wasteful copy, but doesn't dominate.
  best_index.CopyToVec(&best_index_cpu);


  double tot_weight = 0.0,
      tot_accuracy = 0.0;

  // note: we expect that in most cases where this code is called,
  // supervision.Type() will be kSparseMatrix.
  switch (supervision.Type()) {
    case kCompressedMatrix: {
      Matrix<BaseFloat> mat;
      supervision.GetMatrix(&mat);
      for (int32 r = 0; r < num_rows; r++) {
        SubVector<BaseFloat> vec(mat, r);
        BaseFloat row_sum = vec.Sum();
        KALDI_ASSERT(row_sum >= 0.0);
        int32 best_index;
        vec.Max(&best_index);  // discard max value.
        tot_weight += row_sum;
        if (best_index == best_index_cpu[r])
          tot_accuracy += row_sum;
      }
      break;

    }
    case kFullMatrix: {
      const Matrix<BaseFloat> &mat = supervision.GetFullMatrix();
      for (int32 r = 0; r < num_rows; r++) {
        SubVector<BaseFloat> vec(mat, r);
        BaseFloat row_sum = vec.Sum();
        KALDI_ASSERT(row_sum >= 0.0);
        int32 best_index;
        vec.Max(&best_index);  // discard max value.
        tot_weight += row_sum;
        if (best_index == best_index_cpu[r])
          tot_accuracy += row_sum;
      }
      break;
    }
    case kSparseMatrix: {
      const SparseMatrix<BaseFloat> &smat = supervision.GetSparseMatrix();
      for (int32 r = 0; r < num_rows; r++) {
        const SparseVector<BaseFloat> &row = smat.Row(r);
        BaseFloat row_sum = row.Sum();
        int32 best_index;
        row.Max(&best_index);
        KALDI_ASSERT(best_index < num_cols);
        tot_weight += row_sum;
        if (best_index == best_index_cpu[r]) {
          tot_accuracy += row_sum;
        }
        if (!pdfid_classes.empty()) {
          int32 ref_pdf_id = best_index, hyp_pdf_id = best_index_cpu[r];
          for (int c = 0; c < pdfid_classes.size(); ++c) {
            // find ref_pdf_id
            if (std::find(pdfid_classes[c].begin(), pdfid_classes[c].end(), ref_pdf_id)
                != pdfid_classes[c].end()) {
              tot_classes_accuracy[2*c+1] += row_sum;
              tot_classes_accuracy[2*pdfid_classes.size()+1] += row_sum;
              // find hyp_pdf_id
              if (std::find(pdfid_classes[c].begin(), pdfid_classes[c].end(), hyp_pdf_id)
                  != pdfid_classes[c].end()) {
                tot_classes_accuracy[2*c] += row_sum;
                tot_classes_accuracy[2*pdfid_classes.size()] += row_sum;
              }
              break;
            }
          }
        }
      }
      break;
    }
    default: KALDI_ERR << "Bad general-matrix type.";
  }
  *tot_weight_out = tot_weight;
  *tot_accuracy_out = tot_accuracy;
}

const SimpleObjectiveInfo* NnetComputeProb::GetObjective(
    const std::string &output_name) const {
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
      iter = objf_info_.find(output_name);
  if (iter != objf_info_.end())
    return &(iter->second);
  else
    return NULL;
}

double NnetComputeProb::GetTotalObjective(double *tot_weight) const {
  double tot_objectives = 0.0;
  *tot_weight = 0.0;
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher>::const_iterator
    iter = objf_info_.begin(), end = objf_info_.end();
  for (; iter != end; ++iter) {
    tot_objectives += iter->second.tot_objective;
    (*tot_weight) += iter->second.tot_weight;
  }
  return tot_objectives;
}

} // namespace nnet3
} // namespace kaldi
