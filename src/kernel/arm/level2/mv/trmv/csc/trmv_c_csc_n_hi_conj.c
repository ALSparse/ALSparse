#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>
#include <memory.h>

static alphasparse_status_t 
trmv_csc_n_hi_conj_unroll4(const ALPHA_Number alpha,
                   const ALPHA_SPMAT_CSC* A,
                   const ALPHA_Number* x,
                   const ALPHA_Number beta,
                   ALPHA_Number* y,
                   ALPHA_INT lrs,
                   ALPHA_INT lre)                                                         
{
    ALPHA_INT m = A->cols;
    for (ALPHA_INT i = lrs; i < lre; i++)
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
        ALPHA_INT row_ind0, row_ind1, row_ind2, row_ind3;
        ALPHA_Number   *A_val = &A->values[pks];
        ALPHA_INT *A_row = &A->row_indx[pks];
        ALPHA_INT pi;
        for (pi = 0; pi < pkl4; pi += 4)
        {
            ALPHA_Number conj0, conj1, conj2, conj3;
            row_ind0 = A_row[pi];
            row_ind1 = A_row[pi + 1];
            row_ind2 = A_row[pi + 2];
            row_ind3 = A_row[pi + 3];
            alpha_conj(conj0, A_val[pi]);
            alpha_conj(conj1, A_val[pi+1]);
            alpha_conj(conj2, A_val[pi+2]);
            alpha_conj(conj3, A_val[pi+3]);
            if (row_ind3 <= i){
		        alpha_madde(tmp0, conj0, x[row_ind0])
                alpha_madde(tmp1, conj1, x[row_ind1]);
		        alpha_madde(tmp2, conj2, x[row_ind2]);
                alpha_madde(tmp3, conj3, x[row_ind3]);
            }else if (row_ind2 <= i){
                alpha_madde(tmp1, A_val[pi], x[row_ind0]);
                alpha_madde(tmp2, conj1, x[row_ind1]);
                alpha_madde(tmp3, conj2, x[row_ind2]);
            }else if (row_ind1 <= i){
                alpha_madde(tmp2, A_val[pi], x[row_ind0]);
                alpha_madde(tmp3, conj1, x[row_ind1]);
            }else if (row_ind0 <= i){
                alpha_madde(tmp3, A_val[pi], x[row_ind0]);
            }
            
        }
        for (; pi < pkl; pi += 1)
        {
            if (A_row[pi] <= i)
            {
                ALPHA_Number conj0;
                alpha_conj(conj0, A_val[pi]);
		        alpha_madde(tmp0, conj0, x[A_row[pi]]);
            }
        }	
        alpha_add(tmp0, tmp0, tmp1);
        alpha_add(tmp2, tmp2, tmp3);
        alpha_add(tmp0, tmp0, tmp2);
        alpha_mul(tmp0, tmp0, alpha);
        alpha_mul(tmp1, beta, y[i]);
        alpha_add(y[i], tmp0, tmp1); 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

static alphasparse_status_t 
trmv_csc_n_hi_conj_omp(const ALPHA_Number alpha,
                const ALPHA_SPMAT_CSC* A,
                const ALPHA_Number* x,
                const ALPHA_Number beta,
                ALPHA_Number* y)                                                         
{
    ALPHA_INT n = A->cols;

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(A->cols_end, n, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();

        ALPHA_INT local_n_s = partition[tid];
        ALPHA_INT local_n_e = partition[tid + 1];
        
        trmv_csc_n_hi_conj_unroll4(alpha,A,x,beta,y,local_n_s,local_n_e);
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
	return trmv_csc_n_hi_conj_omp(alpha, A, x, beta, y);
}
