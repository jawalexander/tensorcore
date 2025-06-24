// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: matrix

#pragma once

#include <random>

#include "common.h"

class Matrix {
public:
  Matrix(size_t row, size_t col, const std::string &name = "Matrix",
         bool initialized = true, float min = -5.0, float max = 5.0,
         float fix_value = 0.0)
      : m_row(row), m_col(col), m_name(name) {
    HGEMM_CHECK_GT(m_row, 0);
    HGEMM_CHECK_GT(m_col, 0);

    m_elem_num = m_row * m_col;
    HGEMM_CHECK_GT(m_elem_num, 0);

    m_host_ptr = new half[m_elem_num];
    HGEMM_CHECK(m_host_ptr);
    HGEMM_CHECK_CUDART_ERROR(
        cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half)));
    HGEMM_CHECK(m_dev_ptr);

    if (initialized && fix_value >= 0.0) {
      std::random_device rd;
      std::default_random_engine engine{rd()};
      std::uniform_real_distribution<float> uniform(min, max);
      for (size_t i = 0; i < m_elem_num; ++i) {
        m_host_ptr[i] = __float2half(uniform(engine));
        if (i < 10) {
          HLOG("%f", __half2float(m_host_ptr[i]));
        }
      }
    } else {
      memset(m_host_ptr, 0, m_elem_num * sizeof(half));
    }

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyHostToDevice));

    HLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col,
         m_host_ptr, m_dev_ptr);
  }

  ~Matrix() {
    if (m_host_ptr) {
      delete[] m_host_ptr;
      m_host_ptr = nullptr;
    }

    if (m_dev_ptr) {
      HGEMM_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
      m_dev_ptr = nullptr;
    }
  }

  size_t getRow() const { return m_row; }

  size_t getCol() const { return m_col; }

  size_t getElemNum() const { return m_elem_num; }

  half *getHostPtr() const { return m_host_ptr; }

  half *getDevPtr() const { return m_dev_ptr; }

  void tearUp(Matrix *base) {
    HGEMM_CHECK(base);
    HGEMM_CHECK_EQ(m_row, base->getRow());
    HGEMM_CHECK_EQ(m_col, base->getCol());

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, base->getDevPtr(),
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyDeviceToDevice));
  }

  void moveToHost() {
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyDeviceToHost));
  }

  void moveToDevice() {
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr,
                                        m_elem_num * sizeof(half),
                                        cudaMemcpyHostToDevice));
  }

  void memSetHost() { memset(m_host_ptr, 0, m_elem_num * sizeof(half)); }

  void memSetDevice() {
    HGEMM_CHECK_CUDART_ERROR(
        cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half)));
  }

  void checkValue(Matrix *base) {
    HGEMM_CHECK(base);
    HGEMM_CHECK_EQ(m_row, base->getRow());
    HGEMM_CHECK_EQ(m_col, base->getCol());

    m_max_diff = 0.0;
    m_avg_diff = 0.0;
    double diff = 0.0;
    for (size_t i = 0; i < m_elem_num; ++i) {
      diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) -
                                          __half2float(base->getHostPtr()[i])));
      m_max_diff = std::max(m_max_diff, diff);
      m_avg_diff += diff;
    }

    m_avg_diff /= static_cast<double>(m_elem_num);

    HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    int testnum = 0;
    HLOG("test:%d %f %f m_elem_num=%d ", testnum, (float)m_host_ptr[testnum],
         (float)base->getHostPtr()[testnum], m_elem_num);
  }

private:
  const size_t m_row = 0;
  const size_t m_col = 0;
  const std::string m_name = "Matrix";

  size_t m_elem_num = 0;
  half *m_host_ptr = nullptr;
  half *m_dev_ptr = nullptr;

  double m_max_diff = 0.0;
  double m_avg_diff = 0.0;

  HGEMM_DISALLOW_COPY_AND_ASSIGN(Matrix);
};
