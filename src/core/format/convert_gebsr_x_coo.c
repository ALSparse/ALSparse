#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <stdlib.h>

#include "alphasparse/format.h"
#include "alphasparse/util/bitmap.h"
#include <stdio.h>

typedef struct {
  ALPHA_INT row_idx;
  ALPHA_INT col_idx;
  ALPHA_Number value;
} coord_t;

#define ROWS_PER_ROUND (4)
#define NNZ_PADDING_RATIO_BOUND (15.0)
static int cmp_coord(const void *a, const void *b) {
  return ((const coord_t *)a)->col_idx - ((const coord_t *)b)->col_idx;
}
alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_GEBSR **dest,
                          const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim,
                          const alphasparse_layout_t block_layout) {
  alpha_timer_t timer;
  ALPHA_INT m = source->rows;
  ALPHA_INT n = source->cols;
  ALPHA_INT nnz = source->nnz;
  if (m % block_row_dim != 0 || n % block_col_dim != 0) {
    printf("in convert_bsr_coo , rows or cols is not divisible by block_size!!!");
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
  ALPHA_SPMAT_GEBSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_GEBSR));
  *dest = mat;
  ALPHA_INT block_rows = m / block_row_dim;
  ALPHA_INT block_cols = n / block_col_dim;
  ALPHA_INT *block_row_offset =
      alpha_memalign((uint64_t)(block_rows + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
  mat->rows = block_rows;
  mat->cols = block_cols;
  mat->row_block_dim = block_row_dim;
  mat->col_block_dim = block_col_dim;
  mat->block_layout = block_layout;
  mat->rows_start = block_row_offset;
  mat->rows_end = block_row_offset + 1;
  ALPHA_SPMAT_CSR *csr;
  // alpha_timing_start(&timer);
  check_error_return(convert_csr_coo(source, &csr));

  mat->rows_start[0] = 0;
  ALPHA_INT num_threads = alpha_get_thread_num();
  ALPHA_INT block_nnz = 0;

  int blk_sft = 0;
  if (((block_col_dim - 1) & block_col_dim) == 0) {
    int bs = block_col_dim >> 1;
    while (bs) {
      bs >>= 1;
      blk_sft++;
    }
  } else {
    printf("block_size is not power of two\n");
    exit(-1);
  }
  alpha_timing_start(&timer);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    bitmap_t *bitmap;
    bitmap_build(&bitmap, block_cols);
#ifdef _OPENMP
#pragma omp for
#endif
    for (ALPHA_INT row = 0; row < m; row += block_row_dim) {
      const ALPHA_INT start = csr->rows_start[row];
      const ALPHA_INT end = csr->rows_end[row + block_row_dim - 1];
      ALPHA_INT nz_blk_num =
          set_clear_bit_batch_sht_index(bitmap, &csr->col_indx[start], end - start, blk_sft);
      // printf("br %d has %d nnz_block\n", row >> blk_sft, nz_blk_num);
      mat->rows_end[row / block_row_dim] = nz_blk_num;
    }
    bitmap_destory(bitmap);
  }
  alpha_timing_end(&timer);
  double total_time = alpha_timing_elapsed_time(&timer) * 1000;
  // printf("bitmap count nnz block time %f ms\n", total_time);

  // alpha_timing_start(&timer);
  for (ALPHA_INT br = 0; br < block_rows; br++) {
    mat->rows_end[br] = mat->rows_end[br] + mat->rows_end[br - 1];
  }
  // alpha_timing_end(&timer);
  // total_time = alpha_timing_elapsed_time(&timer) * 1000;
  // printf("prefix sum nnz time %f ms\n", total_time);

  block_nnz = mat->rows_end[block_rows - 1];
  const double nnz_padding = 1.0 * block_nnz * block_row_dim * block_col_dim / nnz;
  if (nnz_padding > NNZ_PADDING_RATIO_BOUND) {
    fprintf(stderr,
            "padding too much!!! real nnz counts are:%d, bcsr nnz counts are:%d, ratio is %lf\n",
            nnz, block_nnz * block_row_dim * block_col_dim, nnz_padding);
    mat->rows_start = NULL;
    mat->rows_end = NULL;
    mat->rows = 0;
    mat->cols = 0;
    mat->values = NULL;
    return ALPHA_SPARSE_STATUS_EXECUTION_FAILED;
  }

  mat->col_indx = alpha_memalign((uint64_t)block_nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
  mat->values = alpha_memalign(
      (uint64_t)block_nnz * block_row_dim * block_col_dim * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
  ALPHA_INT *partition = alpha_malloc(sizeof(ALPHA_INT) * (num_threads + 1));
  balanced_partition_row_by_nnz(mat->rows_end, block_rows, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT lrs = partition[tid];
    ALPHA_INT lrh = partition[tid + 1];
    ALPHA_Number *values = &mat->values[mat->rows_start[lrs] * block_row_dim * block_col_dim];
    // count: nnz_block
    ALPHA_INT count = mat->rows_end[lrh - 1] - mat->rows_start[lrs];
    memset(values, '\0', (uint64_t)count * block_row_dim * block_col_dim * sizeof(ALPHA_Number));
    // alpha_timing_start(&timer);
    for (ALPHA_INT br = lrs; br < lrh; br++) {
      ALPHA_Number *values_current_rowblk =
          &mat->values[mat->rows_start[br] * block_row_dim * block_col_dim];
      const ALPHA_INT row_s = br * block_row_dim;
      const ALPHA_INT total_nnz = csr->rows_end[row_s + block_row_dim - 1] - csr->rows_start[row_s];
      if (total_nnz == 0) {
        continue;
      }
      coord_t *points_current_rowblk = (coord_t *)alpha_malloc(sizeof(coord_t) * total_nnz);
      ALPHA_INT *bsr_col_index = &mat->col_indx[mat->rows_start[br]];
      // points_current_rowblk 存储原始矩阵的列坐标 / block_size
      for (ALPHA_INT ir = 0, nnz = 0; ir < block_row_dim; ir++) {
        ALPHA_INT r = br * block_row_dim + ir;
        ALPHA_INT start = csr->rows_start[r];
        ALPHA_INT end = csr->rows_end[r];

        for (ALPHA_INT ai = start; ai < end; ai++) {
          points_current_rowblk[nnz].col_idx = csr->col_indx[ai];
          points_current_rowblk[nnz].row_idx = r;
          points_current_rowblk[nnz].value = csr->values[ai];
          nnz++;
        }
      }
      // printf("\nbefore sort\n");
      // for (ALPHA_INT nnz = 0; nnz < total_nnz; nnz++) {
      //   printf("%d ", points_current_rowblk[nnz].col_idx);
      // }
      // printf("\n");
      qsort(points_current_rowblk, total_nnz, sizeof(coord_t), (__compar_fn_t)cmp_coord);

      ALPHA_INT idx = 0;
      bsr_col_index[idx] = points_current_rowblk[0].col_idx / block_col_dim;
      ALPHA_INT pre = points_current_rowblk[0].col_idx / block_col_dim;
      ALPHA_INT ir = points_current_rowblk[0].row_idx % block_row_dim;
      ALPHA_INT ic = points_current_rowblk[0].col_idx % block_col_dim;
      ALPHA_Number *values_current_blk = values_current_rowblk;

      if (block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
        values_current_blk[ir * block_col_dim + ic] = points_current_rowblk[0].value;
        //  points_current_rowblk存储每个nnz对应 bsr当前行中具体哪一个块
        for (ALPHA_INT nnz = 1; nnz < total_nnz; nnz++) {
          // next blk
          if (pre != points_current_rowblk[nnz].col_idx / block_col_dim) {
            idx++;
            values_current_blk += block_col_dim * block_row_dim;
            bsr_col_index[idx] = points_current_rowblk[nnz].col_idx / block_col_dim;
          }
          pre = points_current_rowblk[nnz].col_idx / block_col_dim;
          ic = points_current_rowblk[nnz].col_idx % block_col_dim;
          ir = points_current_rowblk[nnz].row_idx % block_row_dim;
          values_current_blk[ir * block_col_dim + ic] = points_current_rowblk[nnz].value;
          // #ifdef S
          //           printf("(%d,%d,%f) is inserted to br, blk_idx, ir ic (%d,%d,%d,%d), \n",
          //                  points_current_rowblk[nnz].row_idx,
          //                  points_current_rowblk[nnz].col_idx, points_current_rowblk[nnz].value,
          //                  br, idx, ir, ic);
          // #endif
        }
      } else {
        values_current_blk[ic * block_row_dim + ir] = points_current_rowblk->value;
        //  points_current_rowblk存储每个nnz对应 bsr当前行中具体哪一个块
        for (ALPHA_INT nnz = 1; nnz < total_nnz; nnz++) {
          if (pre != points_current_rowblk[nnz].col_idx / block_col_dim) {
            idx++;
            values_current_blk += block_col_dim * block_row_dim;
            bsr_col_index[idx] = points_current_rowblk[nnz].col_idx / block_col_dim;
          }
          pre = points_current_rowblk[nnz].col_idx / block_col_dim;
          ic = points_current_rowblk[nnz].col_idx % block_col_dim;
          ir = points_current_rowblk[nnz].row_idx % block_row_dim;
          values_current_blk[ic * block_row_dim + ir] = points_current_rowblk[nnz].value;
        }
      }
      const ALPHA_INT block_nnz_br = mat->rows_end[br] - mat->rows_start[br];
      if (idx != block_nnz_br - 1) {
        fprintf(stderr,
                "god, some error occurs, block_nnz of current br %d wrong expected %d, got %d \n",
                br, block_nnz_br, idx + 1);
        exit(-1);
      }
      alpha_free(points_current_rowblk);
    }
  }

// #ifdef S
//   for (int bbr = 0; bbr < mat->rows; bbr++) {
//     int start = mat->rows_start[bbr];
//     int end = mat->rows_end[bbr];
//     printf("br %d\n", bbr);
//     for (int ai = start; ai < end; ai++) {
//       int col = mat->col_indx[ai];
//       printf("\tbc %d\n", col);
//       for (int ir = 0; ir < block_row_dim ; ir++) {
//         for (int ic = 0; ic < block_col_dim; ic++) {
//           ALPHA_Number val = mat->values[ai * block_row_dim * block_col_dim + ir * block_col_dim + ic];
//           if (val) printf("\t\t %d,%d : %f\n", ir, ic, val);
//         }
//       }
//     }
//   }
// #endif

  mat->ordered = true;
  destroy_csr(csr);
  alpha_free(partition);

  mat->d_col_indx = NULL;
  mat->d_rows_ptr = NULL;
  mat->d_values   = NULL;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
