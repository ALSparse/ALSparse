#include "alphasparse/opt.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    ALPHA_Number* diag=(ALPHA_Number*) alpha_malloc(n*sizeof(ALPHA_Number));
    memset(diag, '\0', m * sizeof(ALPHA_Number));

        ALPHA_INT num_thread = alpha_get_thread_num(); 
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT c = 0; c < n; c++)
    {
        for (ALPHA_INT ai = A->cols_start[c]; ai < A->cols_end[c]; ai++)
        {
            ALPHA_INT ar = A->row_indx[ai];
            if (ar == c)
            {
                diag[c] = A->values[ai];
            }
        }
    }
    
    //initialize y as x*alpha
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(int i = 0 ; i < m;i++){
        for(int j = 0 ; j < columns ; j++){
            //initialize y[] as x[]*aplha
            alpha_mul(y[index2(i,j,ldy)], x[index2(i,j,ldx)] ,alpha);
        }
    }

    //csc format, traverse by column
    for(ALPHA_INT c = n - 1; c >= 0;--c){
        //following processing simulates Gaussian Elimination 
        //step 1: processing the lower right diagonal ele such that the coefficient equals to 1
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
        for(ALPHA_INT out_y_col = 0; out_y_col < columns;out_y_col++){
            alpha_div( y[index2(c,out_y_col,ldy)] , y[index2(c,out_y_col,ldy)] ,diag[c]);
        }
        
        for(ALPHA_INT ai = A->cols_end[c]-1; ai >= A->cols_start[c];ai--){
            ALPHA_INT ar = A->row_indx[ai];
            if(ar < c){
        //step 2: use the diagonal ele to eliminate coefficients of other rows at the same column
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
                for(ALPHA_INT out_y_col = 0; out_y_col < columns;out_y_col++){
                    alpha_msube(y[index2(ar,out_y_col,ldy)],A->values[ai],y[index2(c,out_y_col,ldy)]);
                }
            }
        }
        
    }

    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
