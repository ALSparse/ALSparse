#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t
gemv_csc_omp(const ALPHA_Number alpha,
                     const ALPHA_SPMAT_CSC* A,
                     const ALPHA_Number* x,
                     const ALPHA_Number beta,
                     ALPHA_Number* y)
{
	const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	const ALPHA_INT thread_num = alpha_get_thread_num();
	
	ALPHA_INT partition[thread_num + 1];
	balanced_partition_row_by_nnz(A->cols_end, n, thread_num, partition);

	ALPHA_Number** tmp = (ALPHA_Number**)malloc(sizeof(ALPHA_Number*) * thread_num);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
	{
		const ALPHA_INT tid = alpha_get_thread_id();
	
		const ALPHA_INT local_m_s = partition[tid];
		const ALPHA_INT local_m_e = partition[tid + 1];
	    tmp[tid] = (ALPHA_Number*)malloc(sizeof(ALPHA_Number) * m);	
		for (ALPHA_INT k = 0; k < m; k++)
		{
			alpha_setzero(tmp[tid][k]);
		}	
		for (ALPHA_INT i = local_m_s; i < local_m_e; ++i)
    	{
			const ALPHA_Number x_r = x[i];
			int pkl = A->cols_start[i];
			int pke = A->cols_end[i];
			for(;pkl < pke - 3; pkl += 4)
			{
                alpha_madde(tmp[tid][A->row_indx[pkl]], A->values[pkl], x_r);
                alpha_madde(tmp[tid][A->row_indx[pkl + 1]], A->values[pkl + 1], x_r);
                alpha_madde(tmp[tid][A->row_indx[pkl + 2]], A->values[pkl + 2], x_r);
                alpha_madde(tmp[tid][A->row_indx[pkl + 3]], A->values[pkl + 3], x_r);
			}
			for(;pkl < pke; ++pkl)
			{
                alpha_madde(tmp[tid][A->row_indx[pkl]], A->values[pkl], x_r);
			}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for(ALPHA_INT i = 0; i < m; ++i)
	{
		ALPHA_Number tmp_y;
		alpha_setzero(tmp_y);
		for(ALPHA_INT j = 0; j < thread_num; ++j)
		{
            alpha_add(tmp_y, tmp_y, tmp[j][i]);
		}
        alpha_mul(y[i], beta, y[i]);
        alpha_madde(y[i], alpha, tmp_y);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for(ALPHA_INT i = 0; i < thread_num; ++i)
	{
		free(tmp[i]);
	}
	free(tmp);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t 
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_Number beta, ALPHA_Number *y)
{
    return gemv_csc_omp(alpha, A, x, beta, y);
}
