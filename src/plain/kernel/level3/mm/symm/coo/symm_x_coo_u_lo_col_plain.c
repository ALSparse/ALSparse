#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
        {
            ALPHA_Number t;
            alpha_setzero(t);
            alpha_mul(y[i + j * ldy], beta, y[i + j * ldy]);
            alpha_mul(t, alpha, x[i + j * ldx]);
            alpha_add(y[i + j * ldy], y[i + j * ldy], t);
            // y[i] = beta * y[i] + alpha * x[i];
        }
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT ai = 0; ai < mat->nnz; ++ai)
        {
            ALPHA_INT ac = mat->col_indx[ai];
            ALPHA_INT cr = mat->row_indx[ai];
            if (ac < cr)
            {
                ALPHA_Number t;
                alpha_setzero(t);
                alpha_mul(t, mat->values[ai], alpha);
                alpha_madde(y[index2(cc, cr, ldy)], t, x[index2(cc, ac, ldx)]);
                alpha_madde(y[index2(cc, ac, ldy)], t, x[index2(cc, cr, ldx)]);
                // y[index2(cc, cr, ldy)] += alpha * mat->values[ai] * x[index2(cc, ac, ldx)];
                // y[index2(cc, ac, ldy)] += alpha * mat->values[ai] * x[index2(cc, cr, ldx)];
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
