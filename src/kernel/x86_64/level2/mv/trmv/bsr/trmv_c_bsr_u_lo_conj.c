#include "alphasparse/kernel_plain.h"
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

	ALPHA_Number tmp;
	alpha_setzero(tmp);
	for (ALPHA_INT j = 0; j < A->rows  * A->block_size; j++){
		alpha_mul(y[j], y[j], beta);
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
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						alpha_madde(y[m_s+s/bs], alpha, x[col+s/bs]);
						for (int s1 = s; s1 < s +s/bs; s1++){
							tmp.real = 0; tmp.imag = 0;
							ALPHA_Number cv;
							alpha_conj(cv, A->values[s1+ai*bs*bs]);
							alpha_mul(tmp, alpha, cv);
							alpha_mul(tmp, tmp, x[col+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], tmp);
						}
					}
				}else {
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (int s1 = s; s1 < s+bs; s1++){
							tmp.real = 0; tmp.imag = 0;
							ALPHA_Number cv;
							alpha_conj(cv, A->values[s1+ai*bs*bs]);
							alpha_mul(tmp, alpha, cv);
							alpha_mul(tmp, tmp, x[col+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], tmp);
						}
					}
				}if (diag_block == 0){
					for (ALPHA_INT s = 0; s < bs; s++)
						alpha_madde(y[m_s+s], alpha, x[m_s+s]);
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
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first diag indx of the s-row in bolck[ai][col]
						// of A->value
						alpha_madde(y[m_s+s/bs], alpha, x[s/bs+col]);
						for (int s1 = s + 1 + s/bs; s1 < s+bs; s1++){
							tmp.real = 0; tmp.imag = 0;
							ALPHA_Number cv;
							alpha_conj(cv, A->values[s1+ai*bs*bs]);
							alpha_mul(tmp, alpha, cv);
							alpha_mul(tmp, tmp, x[s1-s+col]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], tmp);
						}
					}
				}else {
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (int s1 = s; s1 < s+bs; s1++){
							tmp.real = 0; tmp.imag = 0;
							ALPHA_Number cv;
							alpha_conj(cv, A->values[s1+ai*bs*bs]);
							alpha_mul(tmp, alpha, cv);
							alpha_mul(tmp, tmp, x[s1-s+col]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], tmp);
						}
					}
				}
                if (diag_block == 0){
					for (ALPHA_INT s = 0; s < bs; s++)
						alpha_madde(y[m_s+s], alpha, x[m_s+s]);
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
