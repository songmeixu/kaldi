// nnet3/nnet-binary-component.h

// Copyright 2017 meixu song

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

#ifndef KALDI_NNET3_NNET_BINARY_COMPONENT_H_
#define KALDI_NNET3_NNET_BINARY_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "nnet3/nnet-simple-component.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// Keywords: natural gradient descent, NG-SGD, naturalgradient.  For
/// the top-level of the natural gradient code look here, and also in
/// nnet-precondition-online.h.
/// NaturalGradientAffineComponent is
/// a version of AffineComponent that has a non-(multiple of unit) learning-rate
/// matrix.  See nnet-precondition-online.h for a description of the technique.
/// It is described, under the name Online NG-SGD, in the paper "Parallel
/// training of DNNs with Natural Gradient and Parameter Averaging" (ICLR
/// workshop, 2015) by Daniel Povey, Xiaohui Zhang and Sanjeev Khudanpur.
class BinaryAffineComponent: public AffineComponent {
 public:
  virtual std::string Type() const { return "BinaryAffineComponent"; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  // this constructor does not really initialize, use Init() or Read().
  BinaryAffineComponent() { }
  virtual void Init(int32 input_dim, int32 output_dim,
                    BaseFloat param_stddev, BaseFloat bias_stddev);
  virtual void Init(std::string matrix_filename);
  virtual Component* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  // copy constructor
  explicit BinaryAffineComponent(
      const BinaryAffineComponent &other);

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
 protected:
  virtual void Update(
      const std::string &debug_info,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

 private:
  // disallow assignment operator.
  BinaryAffineComponent &operator= (
      const BinaryAffineComponent&);

  mutable CuMatrix<BaseFloat> w_b;
};

class BinaryActivitionComponent: public NonlinearComponent {
 public:
  explicit BinaryActivitionComponent(const BinaryActivitionComponent &other): NonlinearComponent(other) { }
  BinaryActivitionComponent() { }
  virtual std::string Type() const { return "BinaryActivitionComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kLinearInInput|kPropagateInPlace;
  }
  virtual Component* Copy() const { return new BinaryActivitionComponent(*this); }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
 private:
  BinaryActivitionComponent &operator = (const BinaryActivitionComponent &other); // Disallow.
};

} // namespace nnet3
} // namespace kald

#endif