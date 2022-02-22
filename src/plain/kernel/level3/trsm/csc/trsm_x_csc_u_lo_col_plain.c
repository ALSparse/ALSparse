#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    #ifdef DEBUG
    printf("trsm_csc_u_lo_col_plain called\n");
    #endif
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;

    for(ALPHA_INT out_y_col = 0; out_y_col < columns;out_y_col++){
        for(int i = 0 ; i < n ; i++){
            //initialize y[] as x[]*aplha
            alpha_mul(y[index2(out_y_col,i,ldy)], alpha, x[index2(out_y_col,i,ldx)]);
        }
        //following processing simulates Gaussian Elimination 

        for(ALPHA_INT c = 0; c < n; ++c){//csc format, traverse by column

            for(ALPHA_INT ai = A->cols_start[c]; ai < A->cols_end[c];ai++){
                ALPHA_INT ar = A->row_indx[ai];
                if(c < ar){
                    alpha_msube(y[index2(out_y_col,ar,ldy)], A->values[ai], y[index2(out_y_col,c,ldy)]);
                }
            }
        }
        
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
