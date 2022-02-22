#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>

static alphasparse_status_t
gemv_csr_trans_serial(const ALPHA_Number alpha,
                        const ALPHA_SPMAT_CSR *A,
                        const ALPHA_Number *x,
                        const ALPHA_Number beta,
                        ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    for (ALPHA_INT i = 0; i < m; ++i)
    {
        for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++)
        {
            ALPHA_Number val;
            cmp_conj(val, A->values[ai]);
            alpha_mul(val, alpha, val);
            alpha_madde(y[A->col_indx[ai]], val, x[i]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

static alphasparse_status_t
gemv_csr_trans_omp(const ALPHA_Number alpha,
                     const ALPHA_SPMAT_CSR *A,
                     const ALPHA_Number *x,
                     const ALPHA_Number beta,
                     ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    const ALPHA_INT thread_num = alpha_get_thread_num();
    ALPHA_INT partition[thread_num + 1];
    balanced_partition_row_by_nnz(A->rows_end, m, thread_num, partition);
    ALPHA_Number **tmp = (ALPHA_Number **)malloc(sizeof(ALPHA_Number *) * thread_num);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
    {
        const ALPHA_INT tid = alpha_get_thread_id();
        const ALPHA_INT local_m_s = partition[tid];
        const ALPHA_INT local_m_e = partition[tid + 1];
        tmp[tid] = (ALPHA_Number *)malloc(sizeof(ALPHA_Number) * n);
        memset(tmp[tid],'\0',sizeof(ALPHA_Number) * n);
        for (ALPHA_INT i = local_m_s; i < local_m_e; ++i)
        {
            const ALPHA_Number x_r = x[i];
            int pkl = A->rows_start[i];
            int pke = A->rows_end[i];
            for (; pkl < pke - 3; pkl += 4)
            {
                ALPHA_Number conj0, conj1, conj2, conj3;
                cmp_conj(conj0, A->values[pkl]);
                cmp_conj(conj1, A->values[pkl+1]);
                cmp_conj(conj2, A->values[pkl+2]);
                cmp_conj(conj3, A->values[pkl+3]);
                alpha_madde(tmp[tid][A->col_indx[pkl]], conj0, x_r);
                alpha_madde(tmp[tid][A->col_indx[pkl + 1]], conj1, x_r);
                alpha_madde(tmp[tid][A->col_indx[pkl + 2]], conj2, x_r);
                alpha_madde(tmp[tid][A->col_indx[pkl + 3]], conj3, x_r);
            }
            for (; pkl < pke; ++pkl)
            {
                ALPHA_Number conj0;
                alpha_conj(conj0, A->values[pkl]);
                alpha_madde(tmp[tid][A->col_indx[pkl]], conj0, x_r);
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for (ALPHA_INT i = 0; i < n; ++i)
    {
        ALPHA_Number tmp_y;
        alpha_setzero(tmp_y);
        for (ALPHA_INT j = 0; j < thread_num; ++j)
        {
            alpha_adde(tmp_y, tmp[j][i]);
        }
        alpha_mule(y[i],beta);
        alpha_madde(y[i],alpha,tmp_y);
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for (ALPHA_INT i = 0; i < thread_num; ++i)
    {
        alpha_free(tmp[i]);
    }
    alpha_free(tmp);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
      const ALPHA_SPMAT_CSR *A,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    return gemv_csr_trans_omp(alpha, A, x, beta, y);
}
