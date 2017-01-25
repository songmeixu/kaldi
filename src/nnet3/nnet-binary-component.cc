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

//BinaryAffineComponent::BinaryAffineComponent():
//    max_change_per_sample_(0.0),
//    update_count_(0.0), active_scaling_count_(0.0),
//    max_change_scale_stats_(0.0) { }

void BinaryAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BinaryLinearParams>");
  w_b.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</AffineComponent>");
}

void BinaryAffineComponent::Write(std::ostream &os,
                                                 bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BinaryLinearParams>");
  w_b.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</AffineComponent>");
}

void BinaryAffineComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  w_b.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  w_b.SetZero();
  bias_params_.SetZero();
}

void BinaryAffineComponent::Init(std::string matrix_filename) {
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  w_b.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  w_b.SetZero();
  bias_params_.CopyColFromMat(mat, input_dim);
}

Component* BinaryAffineComponent::Copy() const {
  return new BinaryAffineComponent(*this);
}

void BinaryAffineComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
  w_b.Scale(scale);
}

void BinaryAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const BinaryAffineComponent *other =
      dynamic_cast<const BinaryAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
  w_b.AddMat(alpha, other->w_b);
}

BinaryAffineComponent::BinaryAffineComponent(
    const BinaryAffineComponent &other):
    AffineComponent(other),
    w_b(other.w_b) {
}

void BinaryAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                                     const CuMatrixBase<BaseFloat> &in,
                                                     CuMatrixBase<BaseFloat> *out) const {
  // No need for asserts as they'll happen within the matrix operations.
  w_b.Resize(linear_params_.NumRows(), linear_params_.NumCols());
  w_b.CopyFromMat(Binarize(linear_params_));
  out->AddMatMat(1.0, in, kNoTrans, w_b, kTrans, 1.0);
}

void BinaryAffineComponent::Backprop(const std::string &debug_info,
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

void BinaryAffineComponent::Update(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
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
