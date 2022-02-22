#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT c = 0; c < columns; c++)
        for (ALPHA_INT r = 0; r < mat->rows; r++)
        {
            alpha_mule(y[index2(c, r, ldy)], beta);
            alpha_madde(y[index2(c, r, ldy)], alpha, x[index2(c, r, ldx)]);
        }
    
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT ar = 0; ar < mat->rows; ++ar)
        {
            for (ALPHA_INT ai = mat->rows_start[ar]; ai < mat->rows_end[ar]; ++ai)
            {
                ALPHA_INT ac = mat->col_indx[ai];
                if (ac > ar)
                {
                    ALPHA_Number val;
                    alpha_mul(val, alpha, mat->values[ai]);
                    alpha_madde(y[index2(cc, ar, ldy)], val, x[index2(cc, ac, ldx)]);
                    alpha_madde(y[index2(cc, ac, ldy)], val, x[index2(cc, ar, ldx)]);
                }
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
