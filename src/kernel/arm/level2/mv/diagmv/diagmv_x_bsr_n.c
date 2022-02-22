#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <stdio.h>

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
	if (m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	for (ALPHA_INT j = 0; j < A->rows * A->block_size; j++){
		alpha_mul(y[j], y[j], beta); 
		//y[j] *= beta;
	}
	// For matC, block_layout is defaulted as row_major
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
        ALPHA_INT not_hit_hp = 1;
		for(ALPHA_INT i = 0; i < m_inner; i++){
		ALPHA_INT diag_block = 0;
		 ALPHA_Number temp;
		alpha_setzero(temp);
			for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i];ai++){
				// the block is the diag one
				if(A->col_indx[ai] == i){
					diag_block = 1;
                    not_hit_hp = 0;
					for(ALPHA_INT bi = 0; bi < bs; bi++){
						alpha_mul(temp, x[i*bs+bi], A->values[ai*bs*bs+(bs+1)*bi]);
						alpha_madde(y[i*bs+bi], alpha, temp); 
						//y[i*bs+bi] += alpha*x[i*bs+bi]*A->values[ai*bs*bs+(bs+1)*bi];
					}
				}
			}if (diag_block == 0 && not_hit_hp == 0){
				for (ALPHA_INT s = 0; s < bs; s++){
					y[i*bs+s] = x[i*bs+s];
				}
			}
		}
	}
	// For Fortran, block_layout is defaulted as col_major
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        ALPHA_INT not_hit_hp = 1;
		for(ALPHA_INT i = 0; i < m_inner; i++){
		ALPHA_INT diag_block = 0;
		 ALPHA_Number temp;
		alpha_setzero(temp);
			for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i];ai++){
				// the block is the diag one
				if(A->col_indx[ai] == i){
					diag_block = 1;
                    not_hit_hp = 0;
					for(ALPHA_INT bi = 0; bi < bs; bi++){
						alpha_mul(temp, x[i*bs+bi], A->values[ai*bs*bs+(bs+1)*bi]);
						alpha_madde(y[i*bs+bi], alpha, temp); 
						//y[i*bs+bi] += alpha*x[i*bs+bi]*A->values[ai*bs*bs+(bs+1)*bi];
					}
				}
			}if (diag_block == 0 && not_hit_hp == 0){
				for (ALPHA_INT s = 0; s < bs; s++){
					y[i*bs+s] = x[i*bs+s];
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
