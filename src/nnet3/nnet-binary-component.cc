// nnet3/nnet-simple-component.cc

// Copyright      2017 meixu song

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

#include <iterator>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "nnet3/nnet-binary-component.h";
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

CuMatrix<BaseFloat> Binarize(const CuMatrixBase<BaseFloat> &w) {
  CuMatrix<BaseFloat> w_b(w);
  w_b.ApplyHeaviside();
  w_b.Scale(2.0);
  w_b.Add(-1.0);
  return w_b;
}

//BinaryNaturalGradientAffineComponent::BinaryNaturalGradientAffineComponent():
//    max_change_per_sample_(0.0),
//    update_count_(0.0), active_scaling_count_(0.0),
//    max_change_scale_stats_(0.0) { }

void BinaryNaturalGradientAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BinaryLinearParams>");
  w_b.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<RankIn>");
  ReadBasicType(is, binary, &rank_in_);
  ExpectToken(is, binary, "<RankOut>");
  ReadBasicType(is, binary, &rank_out_);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period_);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history_);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<MaxChangePerSample>");
  ReadBasicType(is, binary, &max_change_per_sample_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<UpdateCount>") {
    ReadBasicType(is, binary, &update_count_);
    ExpectToken(is, binary, "<ActiveScalingCount>");
    ReadBasicType(is, binary, &active_scaling_count_);
    ExpectToken(is, binary, "<MaxChangeScaleStats>");
    ReadBasicType(is, binary, &max_change_scale_stats_);
    ReadToken(is, binary, &token);
  }
  if (token != "<BinaryNaturalGradientAffineComponent>" &&
      token != "</BinaryNaturalGradientAffineComponent>")
    KALDI_ERR << "Expected <BinaryNaturalGradientAffineComponent> or "
              << "</BinaryNaturalGradientAffineComponent>, got " << token;
  SetNaturalGradientConfigs();
}

void BinaryNaturalGradientAffineComponent::Write(std::ostream &os,
                                                 bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BinaryLinearParams>");
  w_b.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, rank_in_);
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, rank_out_);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period_);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history_);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<MaxChangePerSample>");
  WriteBasicType(os, binary, max_change_per_sample_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<UpdateCount>");
  WriteBasicType(os, binary, update_count_);
  WriteToken(os, binary, "<ActiveScalingCount>");
  WriteBasicType(os, binary, active_scaling_count_);
  WriteToken(os, binary, "<MaxChangeScaleStats>");
  WriteBasicType(os, binary, max_change_scale_stats_);
  WriteToken(os, binary, "</BinaryNaturalGradientAffineComponent>");
}

void BinaryNaturalGradientAffineComponent::Init(
    int32 rank_in, int32 rank_out,
    int32 update_period, BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample,
    std::string matrix_filename) {
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  w_b.Resize(output_dim, input_dim);
  w_b.SetZero();
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void BinaryNaturalGradientAffineComponent::Init(
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev, BaseFloat bias_mean,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample) {
  linear_params_.Resize(output_dim, input_dim);
  w_b.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0 &&
      bias_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  w_b.SetZero();
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  bias_params_.Add(bias_mean);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  if (max_change_per_sample > 0.0)
    KALDI_WARN << "You are setting a positive max_change_per_sample for "
               << "NaturalGradientAffineComponent. But it has been deprecated. "
               << "Please use max_change for all updatable components instead "
               << "to activate the per-component max change mechanism.";
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

Component* BinaryNaturalGradientAffineComponent::Copy() const {
  return new BinaryNaturalGradientAffineComponent(*this);
}

void BinaryNaturalGradientAffineComponent::Scale(BaseFloat scale) {
  update_count_ *= scale;
  max_change_scale_stats_ *= scale;
  active_scaling_count_ *= scale;
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
  w_b.Scale(scale);
}

void BinaryNaturalGradientAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const BinaryNaturalGradientAffineComponent *other =
      dynamic_cast<const BinaryNaturalGradientAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  update_count_ += alpha * other->update_count_;
  max_change_scale_stats_ += alpha * other->max_change_scale_stats_;
  active_scaling_count_ += alpha * other->active_scaling_count_;
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
  w_b.AddMat(alpha, other->w_b);
}

BinaryNaturalGradientAffineComponent::BinaryNaturalGradientAffineComponent(
    const BinaryNaturalGradientAffineComponent &other):
    w_b(other.w_b) {
}

void BinaryNaturalGradientAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                                     const CuMatrixBase<BaseFloat> &in,
                                                     CuMatrixBase<BaseFloat> *out) const {
  // No need for asserts as they'll happen within the matrix operations.
  w_b.Resize(linear_params_.NumRows(), linear_params_.NumCols());
  w_b.CopyFromMat(Binarize(linear_params_));
  out->AddMatMat(1.0, in, kNoTrans, w_b, kTrans, 1.0);
}

void BinaryNaturalGradientAffineComponent::Backprop(const std::string &debug_info,
                                                    const ComponentPrecomputedIndexes *indexes,
                                                    const CuMatrixBase<BaseFloat> &in_value,
                                                    const CuMatrixBase<BaseFloat> &,
                                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                                    Component *to_update_in,
                                                    CuMatrixBase<BaseFloat> *in_deriv) const {
  AffineComponent *to_update = dynamic_cast<AffineComponent*>(to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, w_b, kNoTrans, 1.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else {  // the call below is to a virtual function that may be re-implemented
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
    }
  }
}

void BinaryNaturalGradientAffineComponent::Update(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  this->NaturalGradientAffineComponent::Update(debug_info, in_value, out_deriv);
  linear_params_.ApplyCeiling(1.0);
  linear_params_.ApplyFloor(-1.0);
}

void BinaryActivitionComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                          const CuMatrixBase<BaseFloat> &in,
                                          CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(Binarize(in));
}

void BinaryActivitionComponent::Backprop(const std::string &debug_info,
                                         const ComponentPrecomputedIndexes *indexes,
                                         const CuMatrixBase<BaseFloat> &,
                                         const CuMatrixBase<BaseFloat> &,
                                         const CuMatrixBase<BaseFloat> &out_deriv,
                                         Component *to_update, // may be NULL; may be identical
    // to "this" or different.
                                         CuMatrixBase<BaseFloat> *in_deriv) const {
  in_deriv->CopyFromMat(out_deriv);
  in_deriv->CancelGradient();
}

} // namespace nnet3
} // namespace kald
