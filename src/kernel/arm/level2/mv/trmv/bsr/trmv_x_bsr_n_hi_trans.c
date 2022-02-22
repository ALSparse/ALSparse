#include "alphasparse/kernel.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "alphasparse/opt.h"
#include <string.h>
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
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		for (ALPHA_INT i = local_m_s; i < local_m_e; i++){
			ALPHA_INT col = i*bs;
			ALPHA_INT block_start = A->rows_start[i], block_end = A->rows_end[i];
			ALPHA_INT upper_start = alpha_lower_bound(&A->col_indx[block_start], &A->col_indx[block_end], i) - A->col_indx;
			for (ALPHA_INT ai = upper_start; ai < block_end; ai++){
				ALPHA_INT row = A->col_indx[ai];
				ALPHA_INT m_s = row*bs;
				if (row == i){
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s + s/bs; st < s+bs; st++){
							alpha_madde(tmp[tid][m_s+st-s], A->values[st+ai*bs*bs], x[col+s/bs]);
						}
					}
				}else{
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st < s+bs; st++){
							alpha_madde(tmp[tid][m_s+st-s], A->values[st+ai*bs*bs], x[col+s/bs]);
						}
					}
				}
			}
		}
	}else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for (ALPHA_INT i = local_m_s; i < local_m_e; i++){
			ALPHA_INT col = i*bs;
			ALPHA_INT block_start = A->rows_start[i], block_end = A->rows_end[i];
			ALPHA_INT upper_start = alpha_lower_bound(&A->col_indx[block_start], &A->col_indx[block_end], i) - A->col_indx;
			for (ALPHA_INT ai = upper_start; ai < block_end; ai++){
				ALPHA_INT row = A->col_indx[ai];
				ALPHA_INT m_s = row*bs;
				if (row < i){
					continue;
				}else if (row == i){
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st <= s+s/bs; st++){
							alpha_madde(tmp[tid][m_s+s/bs], A->values[st+ai*bs*bs], x[col+st-s]);
						}
					}
				}else{
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st < s+bs; st++){
							alpha_madde(tmp[tid][m_s+s/bs], A->values[st+ai*bs*bs], x[col+st-s]);
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