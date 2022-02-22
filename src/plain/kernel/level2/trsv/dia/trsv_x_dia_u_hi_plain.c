#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_INT main_diag_pos = 0;
    
    for (ALPHA_INT i = 0; i < A->ndiag; i++)
        if(A->distance[i] == 0)
        {
            main_diag_pos = i;
            break;
        }

    for (ALPHA_INT r = A->rows - 1; r >= 0; r--)
    {
        ALPHA_Number temp;
        alpha_setzero(temp);
        for (ALPHA_INT ndiag = main_diag_pos + 1; ndiag < A->ndiag; ndiag++)
        {
            if (A->rows - A->distance[ndiag] > r)
            {
                ALPHA_INT ac = r + A->distance[ndiag];
                alpha_madde(temp, A->values[ndiag * A->lval + r], y[ac]);
                // temp += A->values[ndiag * A->lval + r] * y[ac];
            }
        }
        ALPHA_Number t;
        alpha_setzero(t);
        alpha_mul(t, alpha, x[r]);
        alpha_sub(y[r], t, temp);
        // y[r] = alpha * x[r] - temp;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
