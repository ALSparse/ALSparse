#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {        
        ALPHA_Number temp;// = {.real = 0.0, .imag = 0.0};
        alpha_setzero(temp);
        for (ALPHA_INT cr = 0; cr < A->nnz; cr++)
        {
            int row = A->row_indx[cr];
            int col = A->col_indx[cr];
            if(row == r && col < r)
            {
                alpha_madde(temp, A->values[cr], y[col]);
                // temp += A->values[cr] * y[col];
            }
        }
        ALPHA_Number t; // = {.real = 0.0, .imag = 0.0};
        alpha_mul(t, alpha, x[r]);
        alpha_sub(y[r], t, temp);
        // y[r] = (alpha * x[r] - temp) ;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}