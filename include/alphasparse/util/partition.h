#pragma once

/**
 * @brief header for data partition utils
 */

#include <stdint.h>
#include "alphasparse/types.h"
#include "alphasparse/spmat.h"

#define cross_block_low(id, p, n) ((id) * (n) / (p))
#define cross_block_high(id, p, n) cross_block_low((id) + 1, p, n)
#define cross_block_size(id, p, n) \
  (cross_block_low((id) + 1, p, n) - (cross_block_low(id, p, n)))
#define cross_block_owner(index, p, n) (((p) * ((index) + 1) - 1) / (n))

void find_divisors(ALPHA_INT n, ALPHA_INT *divisors, ALPHA_INT *num);
void balanced_divisors2(ALPHA_INT m, ALPHA_INT n, ALPHA_INT num_threads, ALPHA_INT *divm_p, ALPHA_INT *divn_p);

int lower_bound_int(const ALPHA_INT *t, ALPHA_INT l, ALPHA_INT r, ALPHA_INT value);
int lower_bound_int64(const ALPHA_INT64 *t, ALPHA_INT64 l, ALPHA_INT64 r, ALPHA_INT64 value);

void balanced_partition_row_by_nnz(const ALPHA_INT *acc_sum_arr, ALPHA_INT rows, ALPHA_INT num_threads, ALPHA_INT *partition);
void balanced_partition_row_by_flop(const ALPHA_INT64 *acc_sum_arr, ALPHA_INT rows, ALPHA_INT num_threads, ALPHA_INT *partition);


void block_partition(ALPHA_INT *pointerB, ALPHA_INT *pointerE, ALPHA_INT *block_indx, ALPHA_INT block_dim_len, ALPHA_INT another_dim_len, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);

// pos[index(r,bi,ldp)]
void csr_s_col_partition(const spmat_csr_s_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void csr_d_col_partition(const spmat_csr_d_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void csr_c_col_partition(const spmat_csr_c_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void csr_z_col_partition(const spmat_csr_z_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);

// pos[index(c,bi,ldp)]
void csc_s_row_partition(const spmat_csc_s_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void csc_d_row_partition(const spmat_csc_d_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void csc_c_row_partition(const spmat_csc_c_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void csc_z_row_partition(const spmat_csc_z_t *A, ALPHA_INT cs, ALPHA_INT ce, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);

// pos[index(r,bi,ldp)]
void bsr_s_col_partition(const spmat_bsr_s_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void bsr_d_col_partition(const spmat_bsr_d_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void bsr_c_col_partition(const spmat_bsr_c_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);
void bsr_z_col_partition(const spmat_bsr_z_t *A, ALPHA_INT rs, ALPHA_INT re, ALPHA_INT block_size, ALPHA_INT **pos_p, ALPHA_INT *block_num_p, ALPHA_INT *ldp_p);

#ifndef COMPLEX
#ifndef DOUBLE

#define csr_col_partition csr_s_col_partition
#define bsr_col_partition bsr_s_col_partition
#define csc_row_partition csc_s_row_partition

#else

#define csr_col_partition csr_d_col_partition
#define bsr_col_partition bsr_d_col_partition
#define csc_row_partition csc_d_row_partition

#endif
#else
#ifndef DOUBLE

#define csr_col_partition csr_c_col_partition
#define bsr_col_partition bsr_c_col_partition
#define csc_row_partition csc_c_row_partition

#else

#define csr_col_partition csr_z_col_partition
#define bsr_col_partition bsr_z_col_partition
#define csc_row_partition csc_z_row_partition

#endif
#endif