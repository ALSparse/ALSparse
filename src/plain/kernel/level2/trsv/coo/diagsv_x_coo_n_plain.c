#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        for (ALPHA_INT ai = 0; ai < A->nnz; ai++)
        {
            ALPHA_INT ar = A->row_indx[ai];
            ALPHA_INT ac = A->col_indx[ai];
            if (ac == r && ar == r)
            {
                ALPHA_Number t;
                alpha_mul(t, alpha, x[r]);
                alpha_div(y[r], t, A->values[ai]);
                // y[r] = (alpha * x[r]) / A->values[ai];
                break;
            }
        }        
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
