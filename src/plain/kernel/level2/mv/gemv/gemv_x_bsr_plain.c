#include "alphasparse/kernel_plain.h"
#include "stdio.h"
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		         const ALPHA_SPMAT_BSR* A,
		         const ALPHA_Number* x,
		         const ALPHA_Number beta,
		         ALPHA_Number* y)
{
	ALPHA_INT bs = A->block_size;
	ALPHA_INT m_inner = A->rows/bs;
	m_inner = ( m_inner*bs == A->rows)?(m_inner):(m_inner+1);
	ALPHA_INT n_inner = A->cols/bs;
	n_inner = ( n_inner*bs == A->cols)?(n_inner):(n_inner+1);

	// y = y * beta
	for(ALPHA_INT m = 0; m < A->rows; m++){
		alpha_mul(y[m], y[m], beta);
		//y[m] *= beta;
	}
	 ALPHA_Number temp;
	alpha_setzero(temp);
	// For matC, block_layout is defaulted as row_major
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		for(ALPHA_INT i = 0; i < m_inner; i++){
			for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++){
				// block[ai]: [i][A->col_indx[ai]]
				for(ALPHA_INT row_inner = 0; row_inner < bs; row_inner++){
					for(ALPHA_INT col_inner = 0; col_inner < bs; col_inner++){
						alpha_mul(temp, alpha, A->values[ai*bs*bs+row_inner*bs+col_inner]);
						alpha_mul(temp, temp, x[bs*A->col_indx[ai]+col_inner]);
						alpha_add(y[bs*i+row_inner], y[bs*i+row_inner], temp);
						//y[bs*i+row_inner] += alpha*A->values[ai*bs*bs+row_inner*bs+col_inner]*x[bs*A->col_indx[ai]+col_inner];
					}
				// over for block	
				}
			}
		}
	}
	// For Fortran, block_layout is defaulted as col_major
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for(ALPHA_INT i = 0; i < m_inner; i++){
			for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i];ai++){
				// block[ai]: [i][A->cols[ai]]
				for(ALPHA_INT col_inner = 0; col_inner < bs; col_inner++){
					for(ALPHA_INT row_inner = 0; row_inner < bs; row_inner++){
						alpha_mul(temp, alpha, A->values[ai*bs*bs+col_inner*bs+row_inner]);
						alpha_mul(temp, temp, x[bs*A->col_indx[ai]+col_inner]);
						alpha_add(y[bs*i+row_inner], y[bs*i+row_inner], temp);
						//y[bs*i+row_inner] += alpha*A->values[ai*bs*bs+col_inner*bs+row_inner]*x[bs*A->col_indx[ai]+col_inner];
					}
				// over for block	
				}
			}
		}
	}
	else return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
