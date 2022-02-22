#include "alphasparse/kernel_plain.h"
#include <stdio.h>
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		                    const ALPHA_SPMAT_BSR *A,
		                    const ALPHA_Number *x,
		                    const ALPHA_Number beta,
		                    ALPHA_Number *y)
{
	ALPHA_INT bs = A->block_size;
	ALPHA_INT m_inner = A->rows/bs;
	m_inner = ( m_inner*bs == A->rows)?(m_inner):(m_inner+1);
	ALPHA_INT n_inner = A->cols/bs;
	n_inner = ( n_inner*bs == A->cols)?(n_inner):(n_inner+1);
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	 ALPHA_Number tmp;
	alpha_setzero(tmp);
	for (ALPHA_INT j = 0; j < A->rows; j++){
		alpha_mul(y[j], y[j], beta);
		//y[j] *= beta;
	}
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		ALPHA_INT diag_block = 0;
		for (ALPHA_INT i = 0; i < m_inner; i++){
			ALPHA_INT col = i*bs;
			for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++){	
				ALPHA_INT row = A->col_indx[ai];
				ALPHA_INT m_s = row*bs;
				if (row < i){
					continue;
				}else if (row == i){
					diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						alpha_madde(y[m_s+s/bs], alpha, x[col+s/bs]);
						//y[m_s+s/bs] += alpha*x[col+s/bs];
						for(ALPHA_INT st = s + s/bs + 1; st < s+bs; st++){
							alpha_mul(tmp, alpha, A->values[st+ai*bs*bs]);
							alpha_mul(tmp, tmp, x[col+s/bs]);
							alpha_add(y[m_s+st-s], y[m_s+st-s], tmp);
							//y[m_s+st-s] += alpha*A->values[st+ai*bs*bs]*x[col+s/bs];
						}
					}
				}else{
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st < s+bs; st++){
							alpha_mul(tmp, alpha, A->values[st+ai*bs*bs]);
							alpha_mul(tmp, tmp, x[col+s/bs]);
							alpha_add(y[m_s+st-s], y[m_s+st-s], tmp);
							//y[m_s+st-s] += alpha*A->values[st+ai*bs*bs]*x[col+s/bs];
						}
					}
				}
			}
			if (diag_block == 0){
				for (ALPHA_INT s = 0; s < bs; s++){
					alpha_madde(y[i*bs+s], alpha, x[i*bs+s]);
					//y[i*bs+s] += alpha*x[i*bs+s];
				}
			}
		}
	}else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        ALPHA_INT diag_block = 0;
		for (ALPHA_INT i = 0; i < m_inner; i++){
			ALPHA_INT col = i*bs;
			for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++){
				ALPHA_INT row = A->col_indx[ai];
				ALPHA_INT m_s = row*bs;
				if (row < i){
					continue;
				}else if (row == i){
					diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						alpha_madde(y[m_s+s/bs], alpha, x[col+s/bs]);
						//y[m_s+s/bs] += alpha*x[col+s/bs];
						for(ALPHA_INT st = s; st < s+s/bs; st++){
							alpha_mul(tmp, alpha, A->values[st+ai*bs*bs]);
							alpha_mul(tmp, tmp, x[col+st-s]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*A->values[st+ai*bs*bs]*x[col+st-s];
						}
					}
				}else{
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st < s+bs; st++){
							alpha_mul(tmp, alpha, A->values[st+ai*bs*bs]);
							alpha_mul(tmp, tmp, x[col+st-s]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*A->values[st+ai*bs*bs]*x[col+st-s];
						}
					}
				}
			}
            if (diag_block == 0){
					for (ALPHA_INT s = 0; s < bs; s++)
						alpha_madde(y[i*bs+s], alpha, x[i*bs+s]);
						//y[i*bs+s] += alpha*x[i*bs+s];
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
