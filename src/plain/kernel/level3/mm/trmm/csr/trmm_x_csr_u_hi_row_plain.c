#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT r = 0; r < mat->rows; ++r)
    {
        ALPHA_Number *Y = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < columns; c++)
        {
            ALPHA_Number ctmp;
            alpha_mul(Y[c], Y[c], beta);
            alpha_mul(ctmp, x[index2(r, c, ldx)], alpha);
            alpha_add(Y[c], ctmp, Y[c]);
        }
        for (ALPHA_INT ai = mat->rows_start[r]; ai < mat->rows_end[r]; ai++)
        {
            ALPHA_INT ac = mat->col_indx[ai];
            if (ac > r)
            {
                ALPHA_Number val;
                alpha_mul(val, alpha, mat->values[ai]);
                const ALPHA_Number *X = &x[index2(ac, 0, ldx)];
                for (ALPHA_INT c = 0; c < columns; ++c)
                    alpha_madde(Y[c], X[c], val);
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
