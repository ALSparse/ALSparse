#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
#ifdef DEBUG
    printf("kernel diagsm_bsr_n_col_plain called \n");
#endif
    const ALPHA_INT bs = A->block_size;
    ALPHA_Number* diag=(ALPHA_Number*) alpha_malloc(A->rows*bs*sizeof(ALPHA_Number));
    const ALPHA_INT m = A->rows * bs;
    const ALPHA_INT n = A->cols * bs;
    // assert(m==n);
    memset(diag, '\0', m * sizeof(ALPHA_Number));
    
    const ALPHA_INT b_rows = A->rows;
    const ALPHA_INT b_cols = A->cols;

    for(ALPHA_INT r = 0 ; r < b_rows; r++){
        for(ALPHA_INT ai = A->rows_start[r]; ai < A->rows_end[r]; ai++){
            
            ALPHA_INT ac = A->col_indx[ai];
            if(ac == r){
                for(ALPHA_INT b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(r,b_row,bs)] = A->values[ai * bs * bs +  b_row *(bs + 1)];

                }
            }
        }
    }
    
    for (ALPHA_INT c = 0; c < columns; ++c)
    {
        for (ALPHA_INT r = 0; r < A->rows * bs; ++r)
        {
            ALPHA_Number t;
            alpha_mul(t, alpha, x[index2(c, r, ldx)]);
            alpha_div(y[index2(c, r, ldy)], t, diag[r]);
        }
    }

    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
