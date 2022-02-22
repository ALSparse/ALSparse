#pragma once

/**
 * @brief header for ict malloc utils
 */

#include <stdlib.h>

#include "../types.h"

#define DEFAULT_ALIGNMENT 32

void *alpha_malloc(size_t bytes);

void *alpha_memalign(size_t bytes, size_t alignment);

void alpha_free(void *point);
void alpha_free_dcu(void *point);

#define L1_CACHE_SIZE (64l << 10)
#define L2_CACHE_SIZE (512l << 10)
#define L3_CACHE_SIZE (32l << 20)
void alpha_clear_cache();

void alpha_fill_s(float *arr, const float num, const size_t size);
void alpha_fill_d(double *arr, const double num, const size_t size);
void alpha_fill_c(ALPHA_Complex8 *arr, const ALPHA_Complex8 num, const size_t size);
void alpha_fill_z(ALPHA_Complex16 *arr, const ALPHA_Complex16 num, const size_t size);

void alpha_parallel_fill_s(float *arr, const float num, const size_t size);
void alpha_parallel_fill_d(double *arr, const double num, const size_t size);
void alpha_parallel_fill_c(ALPHA_Complex8 *arr, const ALPHA_Complex8 num,
                         const size_t size);
void alpha_parallel_fill_z(ALPHA_Complex16 *arr, const ALPHA_Complex16 num,
                         const size_t size);

void alpha_fill_random_s(float *arr, unsigned int seed, const size_t size);
void alpha_fill_random_d(double *arr, unsigned int seed, const size_t size);
void alpha_fill_random_c(ALPHA_Complex8 *arr, unsigned int seed, const size_t size);
void alpha_fill_random_z(ALPHA_Complex16 *arr, unsigned int seed,
                       const size_t size);

void alpha_parallel_fill_random_s(float *arr, unsigned int seed,
                                const size_t size);
void alpha_parallel_fill_random_d(double *arr, unsigned int seed,
                                const size_t size);
void alpha_parallel_fill_random_c(ALPHA_Complex8 *arr, unsigned int seed,
                                const size_t size);
void alpha_parallel_fill_random_z(ALPHA_Complex16 *arr, unsigned int seed,
                                const size_t size);

void alpha_fill_random_int(int *arr, unsigned int seed, const size_t size,
                         int upper);

void alpha_fill_random_long(long long *arr, unsigned int seed, const size_t size,
                          long long upper);
