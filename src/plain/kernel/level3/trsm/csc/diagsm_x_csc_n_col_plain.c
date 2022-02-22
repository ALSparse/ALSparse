#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{//assume A is square
    ALPHA_Number* diag=(ALPHA_Number*) alpha_malloc(A->rows*sizeof(ALPHA_Number));

    memset(diag, '\0', A->rows * sizeof(ALPHA_Number));

    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        for (ALPHA_INT ai = A->cols_start[c]; ai < A->cols_end[c]; ai++)
        {
            ALPHA_INT ar = A->row_indx[ai];
            if (ar == c)
            {
                diag[c] = A->values[ai];

                //break;
            }
        }
    }

    for (ALPHA_INT c = 0; c < columns; ++c)
    {
        for (ALPHA_INT r = 0; r < A->rows; ++r)
        {
            ALPHA_Number t;
            alpha_mul(t, alpha, x[index2(c, r, ldx)]);
            alpha_div(y[index2(c, r, ldy)], t, diag[r]);
        }
    }
    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
