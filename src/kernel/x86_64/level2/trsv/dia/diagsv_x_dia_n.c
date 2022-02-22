#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_Number diag[A->rows];

    memset(diag, '\0', A->rows * sizeof(ALPHA_Number));

    for (ALPHA_INT i = 0; i < A->ndiag; i++)
    {
        if(A->distance[i] == 0)
        {
            for (ALPHA_INT r = 0; r < A->rows; r++)
            {
                diag[r] = A->values[i * A->lval + r];
            }
            break;
        }
    }
    
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        ALPHA_Number t;
        alpha_setzero(t);
        alpha_mul(t, alpha, x[r]);
        alpha_div(y[r], t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
