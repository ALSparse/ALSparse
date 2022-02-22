#include "alphasparse/kernel.h"
#ifdef _OPENMP
#include<omp.h>
#endif
#include"alphasparse/opt.h"
#include<string.h>
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		                    const ALPHA_SPMAT_BSR *A,
							const ALPHA_Number *x,
							const ALPHA_Number beta,
							ALPHA_Number *y)
{
 	ALPHA_INT bs = A->block_size;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
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
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		for (ALPHA_INT i = local_m_s; i < local_m_e; i++){
			ALPHA_INT col = i*bs;
			ALPHA_INT block_start = A->rows_start[i], block_end = A->rows_end[i];
			ALPHA_INT lower_end = alpha_upper_bound(&A->col_indx[block_start], &A->col_indx[block_end], i) - A->col_indx;
			for (ALPHA_INT ai = block_start; ai < lower_end; ai++){
				ALPHA_INT row = A->col_indx[ai];
				ALPHA_INT m_s = row*bs;
				if (row == i){
					for (int s = 0; s < bs*bs; s=s+bs){
						for (int s1 = s; s1 < s +s/bs; s1++){
							alpha_madde(tmp[tid][m_s+s1-s], A->values[s1+ai*bs*bs], x[col+s/bs]);
						}
					}
				}else {
					for (int s = 0; s < bs*bs; s=s+bs){
						for (int s1 = s; s1 < s+bs; s1++){
							alpha_madde(tmp[tid][m_s+s1-s], A->values[s1+ai*bs*bs], x[col+s/bs]);
						}
					}
				}
			}
		}
	}else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for (ALPHA_INT i = local_m_s; i < local_m_e; i++){
			ALPHA_INT col = i*bs;
			ALPHA_INT block_start = A->rows_start[i], block_end = A->rows_end[i];
			ALPHA_INT lower_end = alpha_upper_bound(&A->col_indx[block_start], &A->col_indx[block_end], i) - A->col_indx;
			for (ALPHA_INT ai = block_start; ai < lower_end; ai++){
				ALPHA_INT row = A->col_indx[ai];
				ALPHA_INT m_s = row*bs;
				if (row == i){
					for (int s = 0; s < bs*bs; s=s+bs){
						for (int s1 = s + s/bs + 1; s1 < s+bs; s1++){
							alpha_madde(tmp[tid][m_s+s/bs], A->values[s1+ai*bs*bs], x[s1-s+col]);
						}
					}
				}else {
					for (int s = 0; s < bs*bs; s=s+bs){
						for (int s1 = s; s1 < s+bs; s1++){
							alpha_madde(tmp[tid][m_s+s/bs], A->values[s1+ai*bs*bs], x[s1-s+col]);
						}
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
		alpha_madde(y[i], x[i], alpha);
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