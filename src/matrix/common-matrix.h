#ifndef KALDI_COMMON_MATRIX_H_
#define KALDI_COMMON_MATRIX_H_

#include <cstdio>
#include <iostream>
#include <string>
#include <malloc.h>

#define USE_POSIX_MEM_ALLOC
#define pad_zero_bits 128
//#define DNN_CALC_BLOCK 2

namespace FixedPoint {

template<typename T>
class Matrix {
 private:
  T *_data;

  int _cols;
  int _rows;
  int _rcols;
  int _rrows;
 public:

  Matrix() : _data(0), _cols(0), _rows(0), _rcols(0), _rrows(0) { }
  Matrix(Matrix &b) : _cols(b._cols), _rows(b._rows), _rrows(b._rrows), _rcols(b._rcols) {
    if (_rcols > 0 && _rrows > 0) {
      _data =  (T*)memalign(pad_zero_bits / sizeof(char), sizeof(T)*_rrows * _rcols);
      memcpy(_data, b._data, sizeof(T) * _rcols * _rrows);
    }
  }
  virtual ~Matrix() {
    if (_data)
      free(_data);
    _data = 0;
  }

  T *Data() {
    return _data;
  }

  const T *Data() const {
    return _data;
  }

  const int NumRows() const {
    return _rows;
  }

  const int NumCols() const {
    return _cols;
  }

  const int Stride() const {
    return _rcols;
  }

  void reset() {
    if (_data)
      memset(_data, 0, sizeof(T) * _rrows * _rcols);
  }

  T *RowData(const int &idx) {
    return _data + (idx * _rcols);
  }

  const T *RowData(const int &idx) const {
    return _data + (idx * _rcols);
  }

  void Resize(const int &rows, const int &cols) {
    if (_rows == rows && _cols == cols)
      return;
    if (_rrows >= rows && _rcols >= cols) {
      memset(_data, 0, sizeof(T) * _rrows * _rcols);
      _rows = rows;
      _cols = cols;
      return;
    }
    if (_data)
      free(_data);
    _data = NULL;
    _rows = _rrows = rows;
    _cols = _rcols = cols;
    int pad_zero_size = pad_zero_bits / sizeof(T);
    _rcols = (_rcols + pad_zero_size - 1) / pad_zero_size * pad_zero_size;
    if (_rrows * _rcols != 0) {
      _data = (T*)memalign(pad_zero_bits / sizeof(char), sizeof(T)*_rrows * _rcols);
      memset(_data, 0, sizeof(T) * _rrows * _rcols);
    }
  }

  inline T& operator() (int r, int c) {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
        static_cast<UnsignedMatrixIndexT>(_rows) &&
        static_cast<UnsignedMatrixIndexT>(c) <
            static_cast<UnsignedMatrixIndexT>(_cols));
    return *(_data + r * _rcols + c);
  }

  const Matrix<T> &operator=(const Matrix<T> &b) {
    if (this == &b)
      return *this;
    if (_data)
      free(_data);
    _data = NULL;
    _rows = b._rows;
    _cols = b._cols;
    _rrows = b._rrows;
    _rcols = b._rcols;
    int pad_zero_size = pad_zero_bits / sizeof(T);
    _rcols = (_rcols + pad_zero_size - 1) / pad_zero_size * pad_zero_size;
    if (_rrows * _rcols != 0) {
      _data = (T*)memalign(pad_zero_bits / sizeof(char), sizeof(T)*_rrows * _rcols);
      memcpy(_data, b._data, sizeof(T) * _rrows * _rcols);
    }
  }

  void LoadData(T *data) {
    for (int i = 0; i < _rows; ++i) {
      T *psrc = data + i * _cols;
      T *pdes = RowData(i);
      memcpy(pdes, psrc, sizeof(T) * _cols);
    }
  }

  void LoadFeatContext(T *data, int nframenum, int input_vec_size, int left_context, int right_context) {
    int frame_window = left_context + right_context + 1;
    int ndim = input_vec_size * frame_window;
    if (_rows < nframenum || _cols < ndim)
      Resize(nframenum, ndim);
    for (int i = 0; i < nframenum; ++i) {
      T *pdes = RowData(i);
      T *psrc = data + (i - left_context) * input_vec_size;
      memcpy(pdes, psrc, frame_window * input_vec_size * sizeof(T));
    }
  }

  bool writeTxt(const char *txtName) {
    if (_rows < 1 || _cols < 1) {
      return false;
    }
    FILE *fp = fopen(txtName, "w");
    if (fp == NULL) {
      printf("Matrix:: cannot open %s to write\n", txtName);
      return false;
    }
    for (int i = 0; i < _rows; i++) {
      fprintf(fp, "%f", (float) RowData(i)[0]);
      for (int j = 1; j < _cols; j++) {
        fprintf(fp, "\t%f", (float) RowData(i)[j]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
    return true;
  }
};

} // namespace common

#endif
