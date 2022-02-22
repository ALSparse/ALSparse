#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_INT block_rowA = A->rows;
    ALPHA_INT rowA = A->rows * A->block_size;
    ALPHA_Number diag[rowA]; 
    memset(diag, '\0', sizeof(ALPHA_Number) * rowA);
    ALPHA_INT bs = A->block_size;
    
    for (ALPHA_INT ar = 0; ar < block_rowA; ++ar)
    {
        for (ALPHA_INT ai = A->rows_start[ar]; ai < A->rows_end[ar]; ++ai)
        {
            if (A->col_indx[ai] == ar) 
            {
                for(ALPHA_INT block_i = 0; block_i < bs; block_i++)
                {
                    diag[ar*bs+block_i] = A->values[ai*bs*bs + block_i*bs + block_i];
                }
            } 
        }   
    }
    ALPHA_Number tmp;
    for (ALPHA_INT r = 0; r < A->rows * A->block_size; ++r) 
    {
        alpha_mul(tmp, alpha, x[r]); 
        alpha_div(y[r], tmp, diag[r]); 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
