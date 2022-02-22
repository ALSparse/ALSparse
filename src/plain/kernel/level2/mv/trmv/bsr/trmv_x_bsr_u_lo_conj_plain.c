#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		                    const ALPHA_SPMAT_BSR *A,
							const ALPHA_Number *x,
							const ALPHA_Number beta,
							ALPHA_Number *y)
{
#ifdef COMPLEX
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
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		ALPHA_INT diag_block = 0;
		for (ALPHA_INT i = 0; i < m_inner; i++){
			ALPHA_INT col = i*bs;
			for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++){	
				ALPHA_INT row = A->col_indx[ai];
				ALPHA_INT m_s = row*bs;
				if (row > i){
					continue;
				}else if (row == i){
					diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						alpha_madde(y[m_s+s/bs], alpha, x[col+s/bs]);
						//y[m_s+s/bs] += alpha*x[col+s/bs];
						for (ALPHA_INT s1 = s; s1 < s +s/bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][(i+ai)*bs+s/bs]
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_mul(tmp, alpha, val);
							alpha_mul(tmp, tmp, x[col+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], tmp);
							//y[m_s+s1-s] += alpha*A->values[s1+ai*bs*bs]*x[col+s/bs];
						}
					}
				}else {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_mul(tmp, alpha, val);
							alpha_mul(tmp, tmp, x[col+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], tmp);
							//y[m_s+s1-s] += alpha*A->values[s1+ai*bs*bs]*x[col+s/bs];
						}
					}
				}if (diag_block == 0){
					for (ALPHA_INT s = 0; s < bs; s++)
						alpha_madde(y[m_s+s], alpha, x[m_s+s]);
						//y[m_s+s] += alpha*x[m_s+s];
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
				if (row > i ){
					continue;
				}else if (row == i){
                    diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first diag indx of the s-row in bolck[ai][col]
						// of A->value
						alpha_madde(y[m_s+s/bs], alpha, x[s/bs+col]);
						//y[m_s+s/bs] += alpha*x[s/bs+col];	
						for (ALPHA_INT s1 = s + 1 + s/bs; s1 < s+bs; s1++){
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_mul(tmp, alpha, val);
							alpha_mul(tmp, tmp, x[s1-s+col]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*A->values[s1+ai*bs*bs]*x[s1-s+col];
						}
					}
				}else {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][col*bs+s1-ai*bs*bs-s]
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_mul(tmp, alpha, val);
							alpha_mul(tmp, tmp, x[s1-s+col]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*A->values[s1+ai*bs*bs]*x[s1-s+col];
						}
					}
				}
                if (diag_block == 0){
					for (ALPHA_INT s = 0; s < bs; s++)
						alpha_madde(y[m_s+s], alpha, x[m_s+s]);
						//y[m_s+s] += alpha*x[m_s+s];
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    return ALPHA_SPARSE_STATUS_SUCCESS;
#else
	return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
}
