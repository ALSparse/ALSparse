/** 
 * @Author: Zjj
 * @Date: 2020-06-28 18:25:07
 * @LastEditors: Zjj
 * @LastEditTime: 2020-06-29 22:25:24
 */
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Complex alpha,
      const ALPHA_SPMAT_BSR *A,
      const ALPHA_Complex *x,
      const ALPHA_Complex beta,
      ALPHA_Complex *y)
{
#ifdef PRINT
	printf("kernel hermv_c_bsr_u_lo_trans_plain called\n");
#endif
    //TODO faster without invoking transposed 
    const ALPHA_INT m = A->rows * A->block_size;
	const ALPHA_INT n = A->cols * A->block_size;
	const ALPHA_INT bs = A -> block_size;
	const ALPHA_INT bs2=bs * bs;
    // assert(m==n);
	ALPHA_INT b_rows = A->rows ;
//	b_rows = ( b_rows*bs = = A->rows)?(b_rows):(b_rows+1);

	ALPHA_INT b_cols = A->cols;
//	b_cols = ( b_cols*bs = = A->cols)?(b_cols):(b_cols+1);
	
    if(b_rows != b_cols) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	for (ALPHA_INT j = 0; j < A->rows * A->block_size; j++){
		alpha_mul(y[j], y[j], beta);
		alpha_madde(y[j], alpha, x[j]);
		//y[j] *= beta;
	}
	ALPHA_INT a0_idx = -1;
	ALPHA_INT row = -1;
	ALPHA_INT col = -1;
	ALPHA_Complex val_orig ,val_conj;
	ALPHA_Complex temp_orig = {.real=0.0f, .imag=0.0f};
	ALPHA_Complex temp_conj = {.real=0.0f, .imag=0.0f};
	
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		for(ALPHA_INT br = 0 ; br < b_rows; br++){
			row = br * bs;
			
			for(ALPHA_INT ai = A->rows_start[br]; ai < A->rows_end[br]; ++ai){
	            ALPHA_INT bc = A->col_indx[ai];
				col = bc * bs;
				//block (br,bc)
				if(bc > br ){
					continue;
				}
				a0_idx = ai * bs2;
				// diagonal block containing diagonal entry
				if(bc == br){
					for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col) is unit
						//y[b_row + row] += alpha*A->values[a0_idx + (b_row + 1) * bs]*x[col + b_col];
						
						for(ALPHA_INT b_col = 0; b_col < b_row; b_col++){
							
							val_orig = A->values[a0_idx + b_row * bs + b_col];
							val_conj.real = val_orig.real;
							val_conj.imag = - val_orig.imag;
							alpha_mul(temp_orig, alpha, val_orig);
							alpha_mul(temp_conj, alpha, val_conj);

							alpha_madde(y[b_row + row], temp_conj , x[col + b_col]);
							alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);

						}
					}
				}
				else{
					for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						for(ALPHA_INT b_col = 0; b_col < bs; b_col++){
							
							val_orig = A->values[a0_idx + b_row * bs + b_col];
							val_conj.real = val_orig.real;
							val_conj.imag = - val_orig.imag;
							alpha_mul(temp_orig, alpha, val_orig);
							alpha_mul(temp_conj, alpha, val_conj);

							alpha_madde(y[b_row + row], temp_conj , x[col + b_col]);
							alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);

						}
					}
				}
			}
		}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
    	for(ALPHA_INT br = 0 ; br < b_rows; br++){
			row = br * bs;
			
			for(ALPHA_INT ai = A->rows_start[br]; ai < A->rows_end[br]; ++ai){
	            ALPHA_INT bc = A->col_indx[ai];
				col = bc * bs;
				//block (br,bc)
				if(bc > br ){
					continue;
				}
				a0_idx = ai * bs2;
				// diagonal block containing diagonal entry
				if(bc == br){
					for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						//dignaol entry A(row+b_row,col+b_col) is unit
						//y[b_row + row] += alpha*A->values[a0_idx + (b_row + 1) * bs]*x[col + b_col];
						for(ALPHA_INT b_row = b_col + 1; b_row < bs; b_row++){
							
							val_orig = A->values[a0_idx + b_col * bs + b_row];
							val_conj.real = val_orig.real;
							val_conj.imag = - val_orig.imag;
							alpha_mul(temp_orig, alpha, val_orig);
							alpha_mul(temp_conj, alpha, val_conj);

							alpha_madde(y[b_row + row], temp_conj , x[col + b_col]);
							alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);

						}
					}
				}
				else{
					for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
							
							val_orig = A->values[a0_idx + b_col * bs + b_row];
							val_conj.real = val_orig.real;
							val_conj.imag = - val_orig.imag;
							alpha_mul(temp_orig, alpha, val_orig);
							alpha_mul(temp_conj, alpha, val_conj);

							alpha_madde(y[b_row + row], temp_conj , x[col + b_col]);
							alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);

						}
					}
					
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
