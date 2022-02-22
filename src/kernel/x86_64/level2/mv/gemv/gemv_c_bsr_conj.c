#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"                                                                    
#ifdef _OPENMP
#include <omp.h>
#include <string.h>
#endif
static alphasparse_status_t
gemv_bsr_conj_omp(const ALPHA_Number alpha,
                       const ALPHA_SPMAT_BSR* A,
                       const ALPHA_Number* x,
                       const ALPHA_Number beta,
                       ALPHA_Number* y)
{
	ALPHA_INT bs = A->block_size;
    ALPHA_INT bs2 = bs * bs;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
    const ALPHA_INT thread_num = alpha_get_thread_num();
    ALPHA_INT partition[thread_num + 1];
    balanced_partition_row_by_nnz(A->rows_end, m_inner, thread_num, partition);
    ALPHA_Number** tmp = (ALPHA_Number**)malloc(sizeof(ALPHA_Number*) * thread_num);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
    {
        const ALPHA_INT tid = alpha_get_thread_id();
        const ALPHA_INT local_m_s = partition[tid];
        const ALPHA_INT local_m_e = partition[tid + 1];
        tmp[tid] = (ALPHA_Number*)malloc(sizeof(ALPHA_Number)*n_inner*bs);
        memset(tmp[tid], 0, sizeof(ALPHA_Number)*n_inner*bs);
      	
		if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
			for (ALPHA_INT i = local_m_s; i < local_m_e; i++)
			{
				for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i];ai++)
				{
					// A index is (bs * i + block_row, bs * A->col_indx[ai] + block_col)
					// should multiplied by x[bs * i + block_row], 
					for(ALPHA_INT block_row = 0; block_row < bs; block_row++){
						for(ALPHA_INT block_col = 0; block_col < bs; block_col++){
							ALPHA_Number cv = A->values[ai*bs2+block_col+block_row*bs];
							alpha_conj(cv, cv);
							alpha_madde(tmp[tid][bs*A->col_indx[ai]+block_col], cv, x[bs*i+block_row]);
						}
					}
				}    
			}
      	}
	  	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
			for (ALPHA_INT i = local_m_s; i < local_m_e; i++)
			{
				for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i];ai++)
				{
					// index is (bs * i + block_row, bs * A->col_indx[ai] + block_col)
					// should multiplied by x[bs * i + block_row], 
					for(ALPHA_INT block_col = 0; block_col < bs; block_col++){
						for(ALPHA_INT block_row = 0; block_row < bs; block_row++){
							ALPHA_Number cv = A->values[ai*bs2+block_col*bs+block_row];
							alpha_conj(cv, cv);
							alpha_madde(tmp[tid][bs*A->col_indx[ai]+block_col], cv, x[bs*i+ block_row]);
						}
					}
				}    
			}
      	}
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for(ALPHA_INT i = 0; i < n_inner*bs; ++i){
        ALPHA_Number tmp_y;
        alpha_setzero(tmp_y);
        for(ALPHA_INT j = 0; j < thread_num; ++j)
        {
            alpha_add(tmp_y, tmp_y, tmp[j][i]);
        }
        alpha_mul(y[i], y[i], beta);
        alpha_madde(y[i], tmp_y, alpha);
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
                const ALPHA_SPMAT_BSR* A,
                const ALPHA_Number* x,
                const ALPHA_Number beta,
                ALPHA_Number* y)
{
    return gemv_bsr_conj_omp(alpha, A, x, beta, y);
}