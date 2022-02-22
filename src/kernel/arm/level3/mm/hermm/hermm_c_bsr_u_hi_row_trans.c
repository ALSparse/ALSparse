#include"alphasparse/kernel.h"
#include"alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows * mat->block_size; i++)
        for(ALPHA_INT j = 0; j < columns; j++){
            alpha_mul(y[j + i * ldy], y[j + i * ldy], beta);
            alpha_madde(y[j + i * ldy], alpha, x[j + i * ldy]);
        }
            
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
        for(ALPHA_INT row = 0 ; row < m ; row += bs){
            ALPHA_INT br = row / bs;
            for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai){
                const ALPHA_INT bc = mat->col_indx[ai];
                const ALPHA_INT col = bc * bs;
                if(bc < br ){
					continue;
				}
                a0_idx = ai * bs2;
                if(bc == br){
                    for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_col = b_row + 1; b_col < bs; b_col++){
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = mat->values[a0_idx + b_row * bs + b_col];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);
                                alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
                            }
                        }
                        
                    }
                }
                
                else{
                    for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_col = 0; b_col < bs; b_col++){
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = mat->values[a0_idx + b_row * bs + b_col];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);
                                alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
                            }
                        }
                    }
                }
            }
        }
    }
    
    else if( mat->block_layout== ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        for(ALPHA_INT row = 0 ; row < m ; row += bs){
            ALPHA_INT br = row / bs;
            
            for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai){
                const ALPHA_INT bc = mat->col_indx[ai];
                const ALPHA_INT col = bc * bs;
                if(bc < br ){
					continue;
				}
                a0_idx = ai * bs2;
                if(bc == br){
                    for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_row = 0; b_row < b_col; b_row++){
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = mat->values[a0_idx + b_row * bs + b_col];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);

                                alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
                            }
                        }
                    }
                }
                else{
                    for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = mat->values[a0_idx + b_row * bs + b_col];
                                val_conj.real = val_orig.real;
                                val_conj.imag = - val_orig.imag;
                                alpha_mul(temp_orig, alpha, val_orig);
                                alpha_mul(temp_conj, alpha, val_conj);
                                alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
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
