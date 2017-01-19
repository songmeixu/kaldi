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
  CuMatrix<BaseFloat> w_b = w;
  w_b.ApplyHeaviside();
  w_b.Scale(2.0);
  w_b.Add(-1.0);
  return w_b;
}

void BinaryNaturalGradientAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                                     const CuMatrixBase<BaseFloat> &in,
                                                     CuMatrixBase<BaseFloat> *out) const {
  // No need for asserts as they'll happen within the matrix operations.
  w_b = Binarize(linear_params_);
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
