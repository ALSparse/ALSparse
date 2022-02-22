#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t
trmv_csc_u_hi_omp(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_CSC *A,
		              const ALPHA_Number *x,
		              const ALPHA_Number beta,
		              ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	
	const ALPHA_INT thread_num = alpha_get_thread_num();
	
	ALPHA_INT partition[thread_num + 1];
	balanced_partition_row_by_nnz(A->cols_end, n, thread_num, partition);

	ALPHA_Number** tmp = (ALPHA_Number**)malloc(sizeof(ALPHA_Number*) * thread_num);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
    {
		const ALPHA_INT tid = alpha_get_thread_id();
		const ALPHA_INT local_n_s = partition[tid];
		const ALPHA_INT local_n_e = partition[tid + 1];
		tmp[tid] = (ALPHA_Number*)malloc(sizeof(ALPHA_Number) * m);
		for(ALPHA_INT j = 0; j < m; ++j) {
			alpha_setzero(tmp[tid][j]);
		}		
    	for(ALPHA_INT i = local_n_s; i < local_n_e; ++i)
    	{
			const ALPHA_Number x_r = x[i];
			register ALPHA_Number tmp_t;
        alpha_setzero(tmp_t);
			ALPHA_INT cs = A->cols_start[i];
			ALPHA_INT ce = A->cols_end[i];
    	    for(; cs < ce-3; cs += 4)
    	    {
    	        const ALPHA_INT row_0 = A->row_indx[cs];
		    	const ALPHA_INT row_1 = A->row_indx[cs+1];
		   	 	const ALPHA_INT row_2 = A->row_indx[cs+2];
		   	 	const ALPHA_INT row_3 = A->row_indx[cs+3];
    	        if(row_3 < i)
    	        {
					alpha_mul(tmp_t, A->values[cs], x_r);
					alpha_madde(tmp[tid][row_0], alpha, tmp_t);
					alpha_mul(tmp_t, A->values[cs+1], x_r);
					alpha_madde(tmp[tid][row_1], alpha, tmp_t);
					alpha_mul(tmp_t, A->values[cs+2], x_r);
					alpha_madde(tmp[tid][row_2], alpha, tmp_t);
					alpha_mul(tmp_t, A->values[cs+3], x_r);
					alpha_madde(tmp[tid][row_3], alpha, tmp_t);
    	        }else if (row_2 < i){
					alpha_mul(tmp_t, A->values[cs], x_r);
					alpha_madde(tmp[tid][row_0], alpha, tmp_t);
					alpha_mul(tmp_t, A->values[cs+1], x_r);
					alpha_madde(tmp[tid][row_1], alpha, tmp_t);
					alpha_mul(tmp_t, A->values[cs+2], x_r);
					alpha_madde(tmp[tid][row_2], alpha, tmp_t);
		    	}else if (row_1 < i){
					alpha_mul(tmp_t, A->values[cs], x_r);
					alpha_madde(tmp[tid][row_0], alpha, tmp_t);
					alpha_mul(tmp_t, A->values[cs+1], x_r);
					alpha_madde(tmp[tid][row_1], alpha, tmp_t);
		    	}else if (row_0 < i){
					alpha_mul(tmp_t, A->values[cs], x_r);
					alpha_madde(tmp[tid][row_0], alpha, tmp_t);
		    	}
    	    }
			for (;cs < ce;++cs)
			{
		    	const ALPHA_INT row = A->row_indx[cs];
		    	if (row < i){
					alpha_mul(tmp_t, A->values[cs], x_r);
					alpha_madde(tmp[tid][row], alpha, tmp_t);
		    	}
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
		alpha_madde(tmp_y, alpha, x[i]);
		alpha_madde(tmp_y, y[i], beta);
		y[i] = tmp_y;
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
ONAME(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_CSC *A,
					  const ALPHA_Number *x,
					  const ALPHA_Number beta,
					  ALPHA_Number *y)
{
	return trmv_csc_u_hi_omp(alpha, A, x, beta, y);
}
