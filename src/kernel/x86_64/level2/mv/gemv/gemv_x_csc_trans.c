#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>
#include <memory.h>

static alphasparse_status_t 
gemv_csc_trans_omp_1(const ALPHA_Number alpha,
               const ALPHA_SPMAT_CSC* A,
               const ALPHA_Number* x,
               const ALPHA_Number beta,
               ALPHA_Number* y)                                                         
{
    ALPHA_INT m = A->rows;
    ALPHA_INT n = A->cols;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(A->cols_end, n, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
     {
        ALPHA_INT tid = alpha_get_thread_id();

        ALPHA_INT local_m_s = partition[tid];
        ALPHA_INT local_m_e = partition[tid + 1];

        for (ALPHA_INT i = local_m_s; i < local_m_e; i++)
        {
	        register ALPHA_Number tmp0;
            register ALPHA_Number tmp1; 
            register ALPHA_Number tmp2;
            register ALPHA_Number tmp3;
            alpha_setzero(tmp0);
            alpha_setzero(tmp1);
            alpha_setzero(tmp2);
            alpha_setzero(tmp3); 
            ALPHA_INT pks = A->cols_start[i];
            ALPHA_INT pke = A->cols_end[i];
            ALPHA_INT pkl = pke - pks;
            ALPHA_INT pkl4 = pkl - 4;
            ALPHA_INT col_ind0, col_ind1, col_ind2, col_ind3;
            ALPHA_Number *A_val = &A->values[pks];
            ALPHA_INT *A_col = &A->row_indx[pks];
            ALPHA_INT pi;
            for (pi = 0; pi < pkl4; pi += 4)
            {
                col_ind0 = A_col[pi];
                col_ind1 = A_col[pi + 1];
                col_ind2 = A_col[pi + 2];
                col_ind3 = A_col[pi + 3];
                alpha_madde(tmp0, A_val[pi], x[col_ind0]);
                alpha_madde(tmp1, A_val[pi + 1], x[col_ind1]);
                alpha_madde(tmp2, A_val[pi + 2], x[col_ind2]);
                alpha_madde(tmp3, A_val[pi + 3], x[col_ind3]);
            }
            for (; pi < pkl; pi += 1)
            {
                alpha_madde(tmp0, A_val[pi], x[A_col[pi]]);
            }
            alpha_add(tmp0, tmp0, tmp1);
            alpha_add(tmp2, tmp2, tmp3);
            alpha_add(tmp0, tmp0, tmp2);
            alpha_mul(y[i], beta, y[i]);
            alpha_madde(y[i], alpha, tmp0);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		               const ALPHA_SPMAT_CSC *A,
		               const ALPHA_Number *x,
		               const ALPHA_Number beta,
		               ALPHA_Number *y)
{
	return gemv_csc_trans_omp_1(alpha, A, x, beta, y);
}
