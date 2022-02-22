/** 
 * @Author: Zjj
 * @Date: 2020-06-30 23:37:00
 * @LastEditors: Zjj
 * @LastEditTime: 2020-07-01 20:27:47
 */

#include"alphasparse/kernel.h"
#include"alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
#ifdef PRINT
	printf("kernel hermm_c_bsr_n_lo_col_trans_plain called\n");
#endif
    for(ALPHA_INT j = 0; j < columns; j++)
        for (ALPHA_INT i = 0; i < mat->rows * mat->block_size; i++)
            alpha_mul(y[i + j * ldy], y[i + j * ldy], beta);
            
    const ALPHA_INT m = mat->rows * mat->block_size;
    const ALPHA_INT n = mat->cols * mat->block_size;

    const ALPHA_INT bs = mat->block_size;
    const ALPHA_INT bs2 = bs * bs;

    ALPHA_INT a0_idx = -1;
	ALPHA_INT col = -1;
	ALPHA_Complex val_orig ,val_conj;
	ALPHA_Complex temp_orig = {.real=0.0f, .imag=0.0f};
	ALPHA_Complex temp_conj = {.real=0.0f, .imag=0.0f};
    
    if(mat->block_layout== ALPHA_SPARSE_LAYOUT_ROW_MAJOR){

        for(ALPHA_INT c = 0 ; c < columns ; c++){
            for(ALPHA_INT row = 0 ; row < m ; row +=bs ){
                const ALPHA_INT br = row / bs;
                
                for(ALPHA_INT ai= mat->rows_start[br]; ai < mat->rows_end[br]; ++ai){
                    const ALPHA_INT bc = mat->col_indx[ai];
                    const ALPHA_INT col = bc * bs;
                    
                    if(bc > br ){
					    continue;
				    }
                    a0_idx = ai * bs2;
                    if(bc == br){
                        for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col)
						alpha_mul(temp_orig, alpha, mat->values[a0_idx + b_row  * ( bs+1 ) ]);
						alpha_madde(y[index2(c,b_row+row,ldy)], temp_orig , x[index2(c,col + b_row,ldx)]);
                            for(ALPHA_INT b_col = 0; b_col < b_row; b_col++){
                                val_orig = mat->values[a0_idx + b_row * bs + b_col];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);

                                alpha_madde(y[index2(c,b_row + row,ldy)], temp_conj , x[index2(c,col + b_col,ldx)]);
                                alpha_madde(y[index2(c,b_col + col,ldy)], temp_orig , x[index2(c,row + b_row,ldx)]);
                            }
                        }
                    }
                    else{
                        for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
                            for(ALPHA_INT b_col = 0; b_col < bs; b_col++){
                                val_orig = mat->values[a0_idx + b_row * bs + b_col];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);

                                alpha_madde(y[index2(c,b_row + row,ldy)], temp_conj , x[index2(c,col + b_col,ldx)]);
                                alpha_madde(y[index2(c,b_col + col,ldy)], temp_orig , x[index2(c,row + b_row,ldx)]);
                            }
                        }
                    }
                }
            }
        }
    }
    
    else if( mat->block_layout== ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        for(ALPHA_INT c = 0 ; c < columns ; c++){
            for(int row = 0 ; row < m ; row +=bs ){
                const ALPHA_INT br = row / bs;
                
                for(ALPHA_INT ai= mat->rows_start[br]; ai < mat->rows_end[br]; ++ai){
                    
                    const ALPHA_INT bc = mat->col_indx[ai];
                    const ALPHA_INT col = bc * bs;
                    
                    if(bc > br ){
					    continue;
				    }
                    a0_idx = ai * bs2;
                    
                    if(bc == br){
                        for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
                            //dignaol entry A(row+b_row,col+b_col)
                            //y[b_row + row] += alpha*A->values[a0_idx + (b_row + 1) * bs]*x[col + b_col];
                            for(ALPHA_INT b_row = b_col + 1; b_row < bs; b_row++){
                                
                                val_orig = mat->values[a0_idx + b_col * bs + b_row];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);
                                alpha_madde(y[index2(c,b_row + row,ldy)], temp_conj , x[index2(c,col + b_col,ldx)]);
                                alpha_madde(y[index2(c,b_col + col,ldy)], temp_orig , x[index2(c,row + b_row,ldx)]);
                            }
                            alpha_mul(temp_orig, alpha, mat->values[a0_idx + b_col  * ( bs+1 ) ]);
                            alpha_madde(y[b_col + row], temp_orig , x[b_col + col]);
                        }
                    }
                    else{
                        for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
                            for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
                                
                                val_orig = mat->values[a0_idx + b_col * bs + b_row];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);
                                alpha_madde(y[index2(c,b_row + row,ldy)], temp_conj , x[index2(c,col + b_col,ldx)]);
                                alpha_madde(y[index2(c,b_col + col,ldy)], temp_orig , x[index2(c,row + b_row,ldx)]);
                            }
                        }
                        
                    }
                }
            }
        }
    }
    else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
