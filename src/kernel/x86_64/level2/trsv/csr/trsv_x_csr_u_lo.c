#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        ALPHA_Number temp;
        alpha_setzero(temp);
        for (ALPHA_INT ai = A->rows_start[r]; ai < A->rows_end[r]; ai++)
        {
            ALPHA_INT ac = A->col_indx[ai];
            if (ac < r)
            {
                alpha_madde(temp, A->values[ai], y[ac]);
            }
        }
        alpha_mul(y[r], alpha, x[r]);
        alpha_sube(y[r], temp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
