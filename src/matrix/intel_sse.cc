#include "intel_sse.h"

void apply_sigmoid(float *start_a, float *result, const int &cnt) {
  float *end = start_a + cnt;
  while (start_a < end) {
    if (*start_a > 0)
      *result++ = 1.0 / (1 + exp(-1.0 * (*start_a++)));
    else {
      float ex = exp(*start_a++);
      *result++ = ex / (ex + 1.0);
    }
  }
}

void apply_sigmoid_int2uchar(int *start_a, FPAct *result, const int &cnt,
                             const float &mag) {

  int * end = start_a + cnt;
  while(start_a < end) {
    if(*start_a > 0)
      *result++ = round(std::numeric_limits<FPAct>::max() / (1 + exp(-1.0 * (* start_a++*mag))));
    else {
      float ex = exp(*start_a++ *mag);
      *result++ = round(std::numeric_limits<FPAct>::max() * ex / (ex + 1.0));
    }
  }
}

void matrix_times_uchar_char(Matrix<FPWeight> &w,
                             Matrix<FPAct> &act,
                             Matrix<FPBias> &res) {
  res.Resize(act.NumRows(), w.NumRows());
  for (int i = 0; i < w.NumRows(); i++) {
    FPWeight *pw = w.RowData(i);
    for (int j = 0; j < act.NumRows(); j++) {

      FPAct *pact = act.RowData(j);
      FPBias *pres = res.RowData(j);
      vector_product<FPWeight, FPAct, FPBias>(pw, pact, pres[i], w.NumCols());
    }
  }
}

void matrix_times(Matrix<FPWeight> &w,
                  Matrix<FPWeight> &act,
                  Matrix<FPBias> &res) {
  res.Resize(act.NumRows(), w.NumRows());
  for (int i = 0; i < w.NumRows(); i++) {
    FPWeight *pw = w.RowData(i);
    for (int j = 0; j < act.NumRows(); j++) {

      FPWeight *pact = act.RowData(j);
      FPBias *pres = res.RowData(j);
      vector_product<FPWeight, FPWeight, FPBias>(pw, pact, pres[i], w.NumCols());
    }
  }
}

// for SVD [5/20/2014 anhaox]
void matrix_times(Matrix<FPWeight16> &w, const Matrix<FPWeight16> &act, Matrix<FPBias> &res) {
  res.Resize(act.NumRows(), w.NumRows());
  for (int i = 0; i < w.NumRows(); i++) {
    FPWeight16 *pw = w.RowData(i);
    for (int j = 0; j < act.NumRows(); j++) {
      const FPWeight16 *pact = act.RowData(j);
      FPBias *pres = res.RowData(j);
      vector_product<FPWeight16, FPWeight16, FPBias>(pw, pact, pres[i], w.NumCols());
      //vector_product_256<FPWeight16, FPWeight16, FPBias>(pw, pact, pres[i], w.cols());
    }
  }
}

void matrix_plus_vector(Matrix<float> &a, Matrix<float> &b, Matrix<float> &res) {
  res.Resize(a.NumRows(), a.NumCols());
  for (int row = 0; row < a.NumRows(); row++) {
    __m128 *ia = (__m128 *) a.RowData(row);
    const __m128 *ie = (__m128 *) (a.RowData(row) + a.NumCols());
    __m128 *ib = (__m128 *) b.Data();
    __m128 *ires = (__m128 *) res.RowData(row);

    while (ia < ie) {
      *ires = _mm_add_ps(*ia, *ib);
      ia++;
      ib++;
      ires++;
    }
  }
}

void matrix_plus_vector(Matrix<int> &a, const Matrix<int> &b, Matrix<int> &res) {
  res.Resize(a.NumRows(), a.NumCols());
  for (int row = 0; row < a.NumRows(); row++) {
    __m128i *ia = (__m128i *) a.RowData(row);
    const __m128i *ie = (__m128i *) (a.RowData(row) + a.NumCols());
    __m128i *ib = (__m128i *) b.Data();
    __m128i *ires = (__m128i *) res.RowData(row);

    while (ia < ie) {
      *ires = _mm_add_epi32(*ia, *ib);
      ia++;
      ib++;
      ires++;
    }
  }

}

void matrix_plus_vector(Matrix<int> &a, Matrix<int> &b, Matrix<int> &res, const int *calc_pos) {
  res.Resize(a.NumRows(), a.NumCols());
  for (int row = 0; row < a.NumRows(); row++) {
    __m128i *ia = (__m128i *) a.RowData(row);
    const __m128i *ie = (__m128i *) (a.RowData(row) + a.NumCols());
    __m128i *ib = (__m128i *) b.Data();
    __m128i *ires = (__m128i *) res.RowData(row);

    while (ia < ie) {
      *ires = _mm_add_epi32(*ia, *ib);
      ia++;
      ib++;
      ires++;
    }
  }
}


void matrix_plus_vector(Matrix<int> &a, Matrix<int> &b, Matrix<int> &res, const int *calc_pos, const int nFrameNum) {
  res.Resize(a.NumRows(), a.NumCols());
  __m128i *ia = (__m128i *) a.RowData(nFrameNum);
  const __m128i *ie = (__m128i *) (a.RowData(nFrameNum) + a.NumCols());
  __m128i *ib = (__m128i *) b.Data();
  __m128i *ires = (__m128i *) res.RowData(nFrameNum);

  while (ia < ie) {
    *ires = _mm_add_epi32(*ia, *ib);
    ia++;
    ib++;
    ires++;
  }
}

void matrix_times_uchar_char(Matrix<FPWeight> &w, Matrix<FPAct> &act, Matrix<FPBias> &res, const int *calc_pos) {
  //res.resize( act.rows(), w.rows() ); // do not fresh the _result_fp32, because the result of last calc may still be there
  for (int i = 0; i < w.NumRows(); i++) {
    if (calc_pos[i] <= 0) continue;
    FPWeight *pw = w.RowData(i);
    for (int j = 0; j < act.NumRows(); j++) {
      FPAct *pact = act.RowData(j);
      FPBias *pres = res.RowData(j);
      if (pres[i] != 0) continue;
      vector_product<FPWeight, FPAct, FPBias>(pw, pact, pres[i], w.NumCols());
    }
  }
}

void matrix_times_uchar_char(Matrix<FPWeight> &w,
                             Matrix<FPAct> &act,
                             Matrix<FPBias> &res,
                             const int *calc_pos,
                             const int nFrameNum) {
  for (int i = 0; i < w.NumRows(); i++) {
    if (calc_pos[i] <= 0) continue;
    FPWeight *pw = w.RowData(i);
    FPAct *pact = act.RowData(nFrameNum);
    FPBias *pres = res.RowData(nFrameNum);
    if (pres[i] != 0) continue;
    vector_product<FPWeight, FPAct, FPBias>(pw, pact, pres[i], w.NumCols());
  }
}

void matrix_times(Matrix<FPWeight> &w, Matrix<FPWeight> &act, Matrix<FPBias> &res, const int *calc_pos) {
  //res.resize( act.rows(), w.rows() ); // do not fresh the _result_fp32, because the result of last calc may still be there
  for (int i = 0; i < w.NumRows(); i++) {
    if (calc_pos[i] <= 0) continue;
    FPWeight *pw = w.RowData(i);
    for (int j = 0; j < act.NumRows(); j++) {
      FPWeight *pact = act.RowData(j);
      FPBias *pres = res.RowData(j);
      if (pres[i] != 0) continue;
      vector_product<FPWeight, FPWeight, FPBias>(pw, pact, pres[i], w.NumCols());
    }
  }
}

void matrix_times(Matrix<FPWeight> &w,
                  Matrix<FPWeight> &act,
                  Matrix<FPBias> &res,
                  const int *calc_pos,
                  const int nFrameNum) {
  for (int i = 0; i < w.NumRows(); i++) {
    if (calc_pos[i] <= 0) continue;
    FPWeight *pw = w.RowData(i);
    FPWeight *pact = act.RowData(nFrameNum);
    FPBias *pres = res.RowData(nFrameNum);
    if (pres[i] != 0) continue;
    vector_product<FPWeight, FPWeight, FPBias>(pw, pact, pres[i], w.NumCols());
  }
}

// for SVD
void matrix_times(Matrix<FPWeight16> &w, Matrix<FPWeight16> &act, Matrix<FPBias> &res, const int *calc_pos) {
  //res.resize( act.rows(), w.rows() ); // do not fresh the _result_fp32, because the result of last calc may still be there
  for (int i = 0; i < w.NumRows(); i++) {
    if (calc_pos[i] <= 0) continue;
    FPWeight16 *pw = w.RowData(i);
    for (int j = 0; j < act.NumRows(); j++) {
      FPWeight16 *pact = act.RowData(j);
      FPBias *pres = res.RowData(j);
      if (pres[i] != 0) continue;
      vector_product<FPWeight16, FPWeight16, FPBias>(pw, pact, pres[i], w.NumCols());
      //vector_product_256<FPWeight16, FPWeight16, FPBias>(pw, pact, pres[i], w.cols());
    }
  }
}

// for SVD
void matrix_times(Matrix<FPWeight16> &w,
                  Matrix<FPWeight16> &act,
                  Matrix<FPBias> &res,
                  const int *calc_pos,
                  const int nFrameNum) {
  for (int i = 0; i < w.NumRows(); i++) {
    if (calc_pos[i] <= 0) continue;
    FPWeight16 *pw = w.RowData(i);
    FPWeight16 *pact = act.RowData(nFrameNum);
    FPBias *pres = res.RowData(nFrameNum);
    if (pres[i] != 0) continue;
    vector_product<FPWeight16, FPWeight16, FPBias>(pw, pact, pres[i], w.NumCols());
    //vector_product_256<FPWeight16, FPWeight16, FPBias>(pw, pact, pres[i], w.cols());
  }
}

void apply_log_softmax_int2float(int *start_a, float *result, const int &cnt,
                                 const float &mag, const int *calc_pos) {
  int *s = start_a;
  float *res = result;
  float max = std::numeric_limits<float>::min();
  int idx = 0;
  while (s < start_a + cnt) {
    if (calc_pos[idx++] > 0) {
      *res = *s * mag;
      if (*res > max)
        max = *res;
    }
    s++;
    res++;
  }
  float sum = 0.0F;
  s = start_a;
  res = result;
  idx = 0;
  while (res < result + cnt) {
    if (calc_pos[idx++] > 0)
      sum += exp(*res = (*res - max));
    res++;
  }
  res = result;
  sum = log(sum);
  res = result;
  idx = 0;
  while (res < result + cnt) {
    if (calc_pos[idx] > 0)
      *res = (*res - sum);
    res++;
    idx++;
  }
}

void apply_log_softmax_int2float(int *start_a, float *result, const int &cnt,
                                 const float &mag) {
  int *s = start_a;
  float *res = result;
  float max = std::numeric_limits<float>::min();
  while (s < start_a + cnt) {
    *res = *s * mag;
    if (*res > max)
      max = *res;
    s++;
    res++;
  }
  float sum = 0.0F;
  s = start_a;
  res = result;
  while (res < result + cnt) {
    sum += exp(*res++ = (*res - max));
  }
  res = result;
  sum = log(sum);
  res = result;
  while (res < result + cnt) {
    *res++ = (*res - sum);
  }
}
