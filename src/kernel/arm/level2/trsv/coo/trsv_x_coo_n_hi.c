#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <string.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_Number diag[A->rows];
    int num_thread = alpha_get_thread_num(); 
    memset(diag, '\0', A->rows * sizeof(ALPHA_Number));

    for (ALPHA_INT r = 0; r < A->nnz; r++)
    {
        if(A->row_indx[r] == A->col_indx[r])
        {
            diag[A->row_indx[r]] = A->values[r];
        }
    }
    
    for (ALPHA_INT r = A->rows - 1; r >= 0; r--)
    {
        ALPHA_Number temp;
        alpha_setzero(temp);
        for (ALPHA_INT cr = A->nnz - 1; cr >= 0; cr--)
        {
            ALPHA_INT row = A->row_indx[cr];
            ALPHA_INT col = A->col_indx[cr];
            if(row == r && col > r)
            {
                alpha_madde(temp, A->values[cr], y[col]);
            }
        }
        ALPHA_Number t; 
        alpha_setzero(t);
        alpha_mul(t, alpha, x[r]);
        alpha_sub(t, t, temp);
        alpha_div(y[r], t, diag[r]);
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}