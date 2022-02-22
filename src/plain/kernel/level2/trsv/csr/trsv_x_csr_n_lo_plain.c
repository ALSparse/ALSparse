#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_Number diag[A->rows];
    memset(diag, '\0', A->rows * sizeof(ALPHA_Number));
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        for (ALPHA_INT ai = A->rows_start[r]; ai < A->rows_end[r]; ai++)
        {
            ALPHA_INT ac = A->col_indx[ai];
            if (ac == r)
            {
                diag[r] = A->values[ai];
            }
        }
    }
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
        ALPHA_Number t;
        alpha_setzero(t);
        alpha_mul(t, alpha, x[r]);
        alpha_sube(t, temp);
        alpha_div(y[r], t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
