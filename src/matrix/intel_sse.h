//
// Created by songmeixu on 16/5/31.
//

#ifndef KALDI_MATMATH_H_
#define KALDI_MATMATH_H_ 1

#include <limits>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include <xmmintrin.h>
#include <mmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>  // for 256-bit calculation

#include "matrix/kaldi-matrix.h"
#include "matrix/common-matrix.h"

namespace FixedPoint {

inline double round(double d) {
  return floor(d + 0.5);
}

typedef unsigned char FPAct;
typedef int8 FPBias;
typedef int8 FPWeight;
typedef int16 FPWeight16;

template<typename T>
T row_abs_max(T *start, const int cnt) {
  T *end = start + cnt;
  T max = 0;
  while (start < end) {
    T myfabs = fabs((float) *start);
    if (myfabs > max)
      max = myfabs;
    start++;
  }
  return max;
}

template<typename T>
T matrix_abs_max(Matrix<T> &mat) {
  T max = 0;
  for (int i = 0; i < mat.NumRows(); i++) {
    T tmp = row_abs_max<T>(mat.RowData(i), mat.NumCols());
    if (tmp > max)
      max = tmp;
  }
  return max;
}

template<typename From, typename To>
void CommonMatrix2KaldiMatrix(const Matrix<From> &from, kaldi::MatrixBase<To> &to) {
  for (int row = 0; row < from.NumRows(); row++) {
    const From *fs = from.RowData(row);
    const From *fe = from.RowData(row) + from.NumCols();
    To *ts = to.RowData(row);
    while (fs < fe) {
      ts[0] = (To) fs[0];
      ts[1] = (To) fs[1];
      ts[2] = (To) fs[2];
      ts[3] = (To) fs[3];
      fs += 4;
      ts += 4;
    }
  }
};

template<typename From, typename To>
void CommonMatrix2KaldiMatrix(const Matrix<From> &from, kaldi::Matrix<To> &to) {
  for (int row = 0; row < from.NumRows(); row++) {
    const From *fs = from.RowData(row);
    const From *fe = from.RowData(row) + from.NumCols();
    To *ts = to.RowData(row);
    while (fs < fe) {
      ts[0] = (To) fs[0];
      ts[1] = (To) fs[1];
      ts[2] = (To) fs[2];
      ts[3] = (To) fs[3];
      fs += 4;
      ts += 4;
    }
  }
};

template<typename From, typename To>
void KaldiMatrix2CommonMatrix(kaldi::Matrix<From> &from, Matrix<To> &to) {
  for (int row = 0; row < from.NumRows(); row++) {
    From *fs = from.RowData(row);
    From *fe = from.RowData(row) + from.NumCols();
    To *ts = to.RowData(row);
    while (fs < fe) {
      ts[0] = (To) fs[0];
      ts[1] = (To) fs[1];
      ts[2] = (To) fs[2];
      ts[3] = (To) fs[3];
      fs += 4;
      ts += 4;
    }
  }
};

template<typename From, typename To>
void linear_quantize(Matrix<From> &from, Matrix<To> &to,
                     float magnitude_a = (std::numeric_limits<From>::max)(),
                     float magnitude_b = (std::numeric_limits<To>::max)()) {
  to.Resize(from.NumRows(), from.NumCols());
  for (int row = 0; row < from.NumRows(); row++) {
    From *fs = from.RowData(row);
    From *fe = from.RowData(row) + from.NumCols();
    To *ts = to.RowData(row);
    while (fs < fe) {
      ts[0] = (To) round(fs[0] / magnitude_a * magnitude_b);
      ts[1] = (To) round(fs[1] / magnitude_a * magnitude_b);
      ts[2] = (To) round(fs[2] / magnitude_a * magnitude_b);
      ts[3] = (To) round(fs[3] / magnitude_a * magnitude_b);
      fs += 4;
      ts += 4;
    }
  }
}

template<typename From, typename To>
void linear_quantize(const kaldi::Matrix<From> &from, Matrix<To> &to,
                     float magnitude_a = (std::numeric_limits<From>::max)(),
                     float magnitude_b = (std::numeric_limits<To>::max)()) {
  to.Resize(from.NumRows(), from.NumCols());
  for (int row = 0; row < from.NumRows(); row++) {
    const From *fs = from.RowData(row);
    const From *fe = from.RowData(row) + from.NumCols();
    To *ts = to.RowData(row);
    while (fs < fe) {
      ts[0] = (To) round(fs[0] / magnitude_a * magnitude_b);
      ts[1] = (To) round(fs[1] / magnitude_a * magnitude_b);
      ts[2] = (To) round(fs[2] / magnitude_a * magnitude_b);
      ts[3] = (To) round(fs[3] / magnitude_a * magnitude_b);
      fs += 4;
      ts += 4;
    }
  }
}


//template<typename From, typename To>
//void linear_quantize(const kaldi::MatrixBase<From> &from, Matrix<To> &to,
//                     float magnitude_a = (std::numeric_limits<From>::max)(),
//                     float magnitude_b = (std::numeric_limits<To>::max)()) {
//  to.Resize(from.NumRows(), from.NumCols());
//  for (int row = 0; row < from.NumRows(); row++) {
//    const From *fs = from.RowData(row);
//    const From *fe = from.RowData(row) + from.NumCols();
//    To *ts = to.RowData(row);
//    // so from.NumCols() must be 4x dims
//    while (fs < fe) {
//      ts[0] = (To) round(fs[0] / magnitude_a * magnitude_b);
//      ts[1] = (To) round(fs[1] / magnitude_a * magnitude_b);
//      ts[2] = (To) round(fs[2] / magnitude_a * magnitude_b);
//      ts[3] = (To) round(fs[3] / magnitude_a * magnitude_b);
//      fs += 4;
//      ts += 4;
//    }
//  }
//}

template<typename From, typename To>
void linear_quantize(const kaldi::MatrixBase<From> &from, Matrix<To> &to,
                     float magnitude_a = (std::numeric_limits<From>::max)(),
                     float magnitude_b = (std::numeric_limits<To>::max)()) {
  to.Resize(from.NumRows(), from.NumCols());
  for (int row = 0; row < from.NumRows(); row++) {
    const From *fs = from.RowData(row);
    const From *fe = from.RowData(row) + from.NumCols();
    To *ts = to.RowData(row);
    while (fs < fe) {
      ts[0] = (To) round(fs[0] / magnitude_a * magnitude_b);
      fs += 1;
      ts += 1;
    }
  }
}

//template<typename From, typename To>
//void linear_quantize(const kaldi::VectorBase<From> &from, Matrix<To> &to,
//                     float magnitude_a = (std::numeric_limits<From>::max)(),
//                     float magnitude_b = (std::numeric_limits<To>::max)()) {
//  to.Resize(1, from.Dim());
//  const From *fs = from.Data();
//  const From *fe = from.Data() + from.Dim();
//  To *ts = to.Data();
//  while (fs < fe) {
//    ts[0] = (To) round(fs[0] / magnitude_a * magnitude_b);
//    ts[1] = (To) round(fs[1] / magnitude_a * magnitude_b);
//    ts[2] = (To) round(fs[2] / magnitude_a * magnitude_b);
//    ts[3] = (To) round(fs[3] / magnitude_a * magnitude_b);
//    fs += 4;
//    ts += 4;
//  }
//}

template<typename From, typename To>
void linear_quantize(const kaldi::VectorBase<From> &from, Matrix<To> &to,
                     float magnitude_a = (std::numeric_limits<From>::max)(),
                     float magnitude_b = (std::numeric_limits<To>::max)()) {
  to.Resize(1, from.Dim());
  const From *fs = from.Data();
  const From *fe = from.Data() + from.Dim();
  To *ts = to.Data();
  while (fs < fe) {
    ts[0] = (To) round(fs[0] / magnitude_a * magnitude_b);
    fs += 1;
    ts += 1;
  }
}

template<typename A, typename B, typename D>
inline void vector_product(const A *start_a, const B *start_b, D &result, const int &cnt);

template<>
inline void vector_product<float>(const float *start_a,
                                  const float *start_b,
                                  float &result,
                                  const int &cnt) {
  const float *end = (start_a + cnt);
  __m128 *as = (__m128 *) (start_a);
  __m128 *bs = (__m128 *) (start_b);
  __m128 m;
  __m128 c = _mm_set_ps1(0.0);

  while (start_a < end) {
    m = _mm_mul_ps(_mm_load_ps(start_a), _mm_load_ps(start_b));
    c = _mm_add_ps(c, m);
    start_a += 4;
    start_b += 4;
  }
  union u {
    __m128 m;
    float f[4];
  } x;
  c = _mm_hadd_ps(c, c);
  c = _mm_hadd_ps(c, c);
  x.m = c;
  result = x.f[0];
}

template<>
inline void vector_product<FPWeight16, FPWeight16, FPWeight16>(const FPWeight16 *start_a,
                                                   const FPWeight16 *start_b,
                                                   FPWeight16 &result,
                                                   const int &cnt) {

  __m128i c;
  __m128i sum = _mm_set_epi32(0, 0, 0, 0);
  __m128i *a = (__m128i *) start_b;
  const __m128i *e = (__m128i *) (start_b + cnt);
  __m128i *b = (__m128i *) start_a;
  while (a < e) {
    c = _mm_mullo_epi32((*a), (*b));
    sum = _mm_add_epi32(c, sum);
    a++;
    b++;
  }
  union u {
    __m128i m;
    int i[4];
  } x;
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  x.m = sum;
  result = x.i[0];
}

template<>
inline void vector_product<FPWeight, FPAct, FPWeight16>(const FPWeight *start_a,
                                                    const FPAct *start_b,
                                                    FPWeight16 &result,
                                                    const int &cnt) {
  __m128i c;
  __m128i sum = _mm_set_epi32(0, 0, 0, 0);
  __m128i lo;
  __m128i hi;
  __m128i *a = (__m128i *) start_b;
  const __m128i *e = (__m128i *) (start_b + cnt);
  __m128i *b = (__m128i *) start_a;

  while (a < e) {
    //c = _mm_maddubs_epi16( _mm_stream_load_si128(a), _mm_stream_load_si128(b) );
    c = _mm_maddubs_epi16((*a), (*b));
    lo = _mm_cvtepi16_epi32(c);
    hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
    sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
    a++;
    b++;
  }
  union u {
    __m128i m;
    int i[4];
  } x;
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  x.m = sum;
  result = x.i[0];
}

template<>
inline void vector_product<FPWeight, FPWeight, FPWeight16>(const FPWeight *start_a,
                                                       const FPWeight *start_b,
                                                       FPWeight16 &result,
                                                       const int &cnt) {
  __m128i c;
  __m128i sum = _mm_set_epi32(0, 0, 0, 0);
  __m128i lo;
  __m128i hi;
  __m128i a_abs;
  __m128i b_sign;
  __m128i *a = (__m128i *) start_b;  // act
  const __m128i *e = (__m128i *) (start_b + cnt);
  __m128i *b = (__m128i *) start_a;  // weight

  while (a < e) {
    //c = _mm_maddubs_epi16( _mm_stream_load_si128(a), _mm_stream_load_si128(b) );
    b_sign = _mm_sign_epi8(*b, *a);
    a_abs = _mm_abs_epi8(*a);
    c = _mm_maddubs_epi16(a_abs, b_sign);
    lo = _mm_cvtepi16_epi32(c);
    hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
    sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
    a++;
    b++;
  }
  union u {
    __m128i m;
    int i[4];
  } x;
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  x.m = sum;
  result = x.i[0];
}

// for SVD
//template<>
//inline void vector_product<FPWeight16, FPWeight16, FPWeight16>(const FPWeight16 *start_a,
//                                                           const FPWeight16 *start_b,
//                                                           FPWeight16 &result,
//                                                           const int &cnt) {
//  __m128i c;
//  __m128i sum = _mm_set_epi32(0, 0, 0, 0);
//  __m128i *a = (__m128i *) start_b;
//  const __m128i *e = (__m128i *) (start_b + cnt);
//  __m128i *b = (__m128i *) start_a;
//
//  while (a < e) {
//    //c = _mm_maddubs_epi16( _mm_stream_load_si128(a), _mm_stream_load_si128(b) );
//    c = _mm_madd_epi16((*a), (*b));
//    sum = _mm_add_epi32(c, sum);
//    a++;
//    b++;
//  }
//  union u {
//    __m128i m;
//    int i[4];
//  } x;
//  sum = _mm_hadd_epi32(sum, sum);
//  sum = _mm_hadd_epi32(sum, sum);
//  x.m = sum;
//  result = x.i[0];
//}

// AVX2 (256-bit) [5/20/2014 anhaox]
/*
template<typename A, typename B, typename D>
inline void vector_product_256(const A * start_a, const B * start_b, D & result, const int & cnt);
template<>
inline void vector_product_256<FPWeight16, FPWeight16, FPWeight16>(const FPWeight16 * start_a, const FPWeight16 * start_b, FPWeight16 & result, const int & cnt)
{
	__m256i c;
	__m256i sum = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
	__m256i * a = (__m256i *)start_b;
	const __m256i * e = (__m256i *)(start_b + cnt);
	__m256i * b = (__m256i *)start_a;

	while(a < e)
	{
		//c = _mm_maddubs_epi16( _mm_stream_load_si128(a), _mm_stream_load_si128(b) );
		c = _mm256_madd_epi16((*a), (*b));
		sum = _mm256_add_epi32(c, sum);
		a++;
		b++;
	}
	union u
	{
		__m256i m;
		int i[8];
	} x;
	sum = _mm256_hadd_epi32(sum, sum);  // [12,34,56,78,12,34,56,78]
	sum = _mm256_hadd_epi32(sum, sum);  // [1234,5678,1234,5678,1234,5678,1234,5678]
	sum = _mm256_hadd_epi32(sum, sum);  // [12345678, ... ,12345678]
	x.m = sum;
	result = x.i[0];
}*/

template<typename T>
void matrix_dot_divide(Matrix<T> &start, const T &divider, Matrix<T> &res) {
  for (int i = 0; i < start.NumRows(); i++) {
    T *start_a = start.RowData(i);
    T *result = res.RowData(i);
    const T *end = start_a + start.NumCols();
    while (start_a < end) {
      *result = *start_a / divider;
      start_a++;
      result++;
    }
  }
}

void matrix_plus_vector(const Matrix<FPWeight16> &a, const Matrix<FPBias> &b, const Matrix<FPWeight16> &res);

void matrix_plus_vector(Matrix<float> &a, Matrix<float> &b, Matrix<float> &res);

void matrix_plus_vector(Matrix<int> &a, const Matrix<int> &b, Matrix<int> &res);

void matrix_plus_vector(Matrix<int> &a, Matrix<int> &b, Matrix<int> &res, const int *calc_pos);

// add for lazy
void matrix_plus_vector(Matrix<int> &a, Matrix<int> &b, Matrix<int> &res, const int *calc_pos, const int nFrameNum);

template<typename T>
void vector_plus(T *start_a, T *start_b, T *result, const int &cnt) {
  T *end = start_a + cnt;
  while (start_a < end) {
    result[0] = start_a[0] + start_b[0];
    result[1] = start_a[1] + start_b[1];
    result[2] = start_a[2] + start_b[2];
    result[3] = start_a[3] + start_b[3];
    start_a += 4;
    start_b += 4;
    result += 4;
  }
}

template<typename T>
void vector_minus(T *start_a, T *start_b, T *result, const int &cnt) {
  T *end = start_a + cnt;
  while (start_a < end) {
    result[0] = start_a[0] - start_b[0];
    result[1] = start_a[1] - start_b[1];
    result[2] = start_a[2] - start_b[2];
    result[3] = start_a[3] - start_b[3];
    start_a += 4;
    start_b += 4;
    result += 4;
  }
}

template<typename T>
void sum(T *start_a, T &result, const int &cnt) {
  T *end = start_a + cnt;
  result = 0;
  while (start_a < end) {
    result += *start_a;
    start_a++;
  }
}

template<typename T>
void apply_scale(T *start_a, const float &scale, T *result, const int &cnt) {
  T *end = start_a + cnt;
  while (start_a < end)
    *result++ = (*start_a++) * scale;
}

template<typename T>
void apply_scale(T *start_a, const float &scale, T *result, const int &cnt, const int *calc_pos) {
  T *end = start_a + cnt;
  int idx = 0;
  while (start_a < end) {
    if (calc_pos[idx++] > 0) *result++ = (*start_a) * scale;
    else *result++ = 0;
    start_a++;
  }
}

template<typename T>
void apply_exp(T *start_a, T *result, const int &cnt) {
  T *end = start_a + cnt;
  while (start_a < end)
    *result++ = exp(*start_a++);
}

template<typename T>
void apply_log(T *start_a, T *result, const int &cnt) {
  T *end = start_a + cnt;
  while (start_a < end)
    *result++ = log(*start_a++);
}

void apply_sigmoid(float *start_a, float *result, const int &cnt);

void apply_sigmoid_int2float(int *start_a, float *result, const int &cnt, const float &mag);

void apply_sigmoid_int2uchar(int *start_a, FPAct *result, const int &cnt, const float &mag);

void matrix_times_uchar_char(Matrix<FPWeight> &w, Matrix<FPAct> &act, Matrix<FPWeight16> &res);

void matrix_times_uchar_char(Matrix<FPWeight> &w, Matrix<FPAct> &act, Matrix<FPWeight16> &res, const int *calc_pos);

void matrix_times_uchar_char(Matrix<FPWeight> &w,
                             Matrix<FPAct> &act,
                             Matrix<FPWeight16> &res,
                             const int *calc_pos,
                             const int nFrameNum);    // work for lazy [10/28/2013 Administrator]

void matrix_times(Matrix<FPWeight> &w, Matrix<FPWeight> &act, Matrix<FPWeight16> &res);

void matrix_times(Matrix<FPWeight> &w, Matrix<FPWeight> &act, Matrix<FPWeight16> &res, const int *calc_pos);

void matrix_times(Matrix<FPWeight> &w, Matrix<FPWeight> &act, Matrix<FPWeight16> &res, const int *calc_pos, const int nFrameNum);

// for SVD [5/20/2014 anhaox]
void matrix_times(Matrix<FPWeight16> &w, const Matrix<FPWeight16> &act, Matrix<FPWeight16> &res);

void matrix_times(Matrix<FPWeight16> &w, Matrix<FPWeight16> &act, Matrix<FPWeight16> &res, const int *calc_pos);

void matrix_times(Matrix<FPWeight16> &w, Matrix<FPWeight16> &act, Matrix<FPWeight16> &res, const int *calc_pos, const int nFrameNum);

template<typename T>
void apply_log_softmax(T *start_a, T *result, const int &cnt) {
  T *s = start_a;
  T *res = result;
  T max = *start_a;
  while (s < start_a + cnt) {
    if (*s > max)
      max = *s;
    s++;
  }
  float sum = 0.0F;
  s = start_a;
  while (s < start_a + cnt) {
    sum += exp(*s++ = (*s - max));
  }
  res = result;
  sum = log(sum);
  s = start_a;
  while (s < start_a + cnt) {
    *res++ = (*s++ - sum);
  }
}

void apply_log_softmax_int2float(int *start_a, float *result, const int &cnt,
                                 const float &mag);

void apply_log_softmax_int2float(int *start_a, float *result, const int &cnt,
                                 const float &mag, const int *calc_pos);

/// attention
/// when calling this, Matrix b should be transposed
/// the result is also transposed
//template<typename T>
//void matrix_times(Matrix<T> &a, Matrix<T> &b, Matrix<T> &res) {
//  if (a._cols != b._cols) {
//    std::cout << "matrix dim not right" << std::endl;
//    return;
//  }
//  res.Resize(b._rows, a._rows);
//  int a_step = a.Stride(), b_step = b.Stride(), res_step = res.Stride();
//  const T *a_start = a.Data(), *b_start = b.Data();
//  T *res_start = res.Data();
//  for (int j = 0; j < b._rows; j++) {
//    const T *a_tmp = a_start;
//    for (int i = 0; i < a._rows; i++) {
//      vector_product<T>(a_tmp, b_start, res_start[i], b._cols);
//      a_tmp += a_step;
//    }
//    b_start += b_step;
//    res_start += res_step;
//  }
//}

template<typename T>
void transpose(Matrix<T> &from, Matrix<T> &to) {
  to.Resize(from.NumCols(), from.NumRows());
  for (int i = 0; i < from.NumRows(); i++)
    for (int j = 0; j < from.NumCols(); j++)
      to.RowData(j)[i] = from.RowData(i)[j];
}

template<typename T>
void matrix_feature_times(Matrix<T> &a, T **feature_set, T *res, const int &feature_size, const int &frame) {
  if (a._cols != frame * feature_size) {
    std::cout << "matrix dim not right" << std::endl;
    return;
  }
  for (int i = 0; i < a._rows; i++) {
    T *m = a.RowData(i);
    res[i] = 0;
    for (int f = 0; f < frame; f++) {
      T tmp;
      vector_product<T>(m, feature_set[f], tmp, feature_size);
      res[i] += tmp;
      m += feature_size;
    }
  }
}

void apply_sigmoid(float *start_a, float *result, const int &cnt);

}

#endif //KALDI_MATMATH_H
