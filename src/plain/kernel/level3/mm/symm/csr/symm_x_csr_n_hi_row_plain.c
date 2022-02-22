#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{

    for (ALPHA_INT r = 0; r < mat->rows; ++r)
        for (ALPHA_INT c = 0; c < columns; c++)
            alpha_mule(y[index2(r, c, ldy)], beta);

    for (ALPHA_INT r = 0; r < mat->rows; ++r)
    {
        for (ALPHA_INT ai = mat->rows_start[r]; ai < mat->rows_end[r]; ai++)
        {
            ALPHA_INT ac = mat->col_indx[ai];
            if (ac > r)
            {
                ALPHA_Number val;
                alpha_mul(val, alpha, mat->values[ai]);
                for (ALPHA_INT c = 0; c < columns; ++c)
                {
                    alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    alpha_madde(y[index2(ac, c, ldy)], val, x[index2(r, c, ldx)]);
                }
            }
            else if (ac == r)
            {
                ALPHA_Number val;
                alpha_mul(val, alpha, mat->values[ai]);
                for (ALPHA_INT c = 0; c < columns; ++c)
                {
                    alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
