#include "alphasparse/opt.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    ALPHA_INT num_thread = alpha_get_thread_num(); 
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif  
    for(ALPHA_INT out_y_col = 0; out_y_col < columns;out_y_col++){
        for(int i = 0 ; i < n ; i++){
            //initialize y[] as x[]*aplha
            alpha_mul(y[index2(out_y_col,i,ldy)], alpha, x[index2(out_y_col,i,ldx)]);
        }

        for(ALPHA_INT c = n - 1; c >= 0;--c){//csc format, traverse by column

            for(ALPHA_INT ai = A->cols_end[c]-1; ai >= A->cols_start[c];ai--){
                
                ALPHA_INT ar = A->row_indx[ai];
                if(ar < c){
                    alpha_msube(y[index2(out_y_col,ar,ldy)], A->values[ai], y[index2(out_y_col,c,ldy)]);
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
