#include "alphasparse/util.h"
#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>
#include <memory.h>

static inline alphasparse_status_t
gemv_csr_unroll4(const ALPHA_Number alpha,
                 const ALPHA_SPMAT_CSR *A,
                 const ALPHA_Number *x,
                 const ALPHA_Number beta,
                 ALPHA_Number *y,
                 ALPHA_INT lrs,
                 ALPHA_INT lre)
{
    for (ALPHA_INT i = lrs; i < lre; i++)
    {
        ALPHA_INT pks = A->rows_start[i];
        ALPHA_INT pke = A->rows_end[i];
        ALPHA_INT pkl = pke - pks;
        ALPHA_Number tmp = vec_doti(pkl, &A->values[pks], &A->col_indx[pks], x);
        alpha_mule(y[i], beta);
        alpha_madde(y[i], alpha, tmp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

static inline alphasparse_status_t
gemv_csr_omp(const ALPHA_Number alpha,
             const ALPHA_SPMAT_CSR *A,
             const ALPHA_Number *x,
             const ALPHA_Number beta,
             ALPHA_Number *y)
{
    ALPHA_INT m = A->rows;

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(A->rows_end, m, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();

        ALPHA_INT local_m_s = partition[tid];
        ALPHA_INT local_m_e = partition[tid + 1];
        gemv_csr_unroll4(alpha, A, x, beta, y, local_m_s, local_m_e);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
      const ALPHA_SPMAT_CSR *mat,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    return gemv_csr_omp(alpha, mat, x, beta, y);
}
