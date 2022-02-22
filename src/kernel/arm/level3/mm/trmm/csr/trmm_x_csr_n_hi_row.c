#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT* acc_nnz = alpha_malloc(sizeof(ALPHA_INT) * mat->rows);
    memset(acc_nnz, '\0', mat->rows * sizeof(ALPHA_INT));
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT ar = 0; ar < mat->rows; ar++)
    {
        for (ALPHA_INT ai = mat->rows_start[ar]; ai < mat->rows_end[ar]; ai++)
        {
            if (mat->col_indx[ai] >= ar)
            {
                acc_nnz[ar] += 1;
            }
        }
    }
    for (ALPHA_INT i = 1; i < mat->rows; i++)
    {
        acc_nnz[i] += acc_nnz[i - 1];
    }
    ALPHA_INT *partition = alpha_malloc((num_threads + 1) * sizeof(ALPHA_INT));
    balanced_partition_row_by_nnz(acc_nnz,mat->rows, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT local_m_s = partition[tid];
        ALPHA_INT local_m_e = partition[tid + 1];
        for (ALPHA_INT r = local_m_s; r < local_m_e; ++r)
        {
            ALPHA_Number *Y = &y[index2(r, 0, ldy)];
            for (ALPHA_INT c = 0; c <columns; c++)
                alpha_mule(Y[c], beta);
            for (ALPHA_INT ai = mat->rows_start[r]; ai < mat->rows_end[r]; ai++)
            {
                ALPHA_INT ac = mat->col_indx[ai];
                if (ac >= r)
                {
                    ALPHA_Number val;
                    alpha_mul(val, alpha, mat->values[ai]);
                    const ALPHA_Number *X = &x[index2(ac, 0, ldx)];
                    for (ALPHA_INT c = 0; c <columns; ++c)
                        alpha_madde(Y[c], val, X[c]);
                }
            }
        }
    }
    alpha_free(partition);
    alpha_free(acc_nnz);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
