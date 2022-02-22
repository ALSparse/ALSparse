#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
        {
            ALPHA_Complex t = {0, 0};
            alpha_mul(y[i + j * ldy], beta, y[i + j * ldy]);
            alpha_mul(t, alpha, x[i + j * ldx]);
            alpha_add(y[i + j * ldy], y[i + j * ldy], t);
            // y[i] = beta * y[i] + alpha * x[i];
        }
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            for (ALPHA_INT ai = mat->rows_start[cr]; ai < mat->rows_end[cr]; ++ai)
            {
                ALPHA_INT ac = mat->col_indx[ai];
                if (ac < cr)
                {
                    ALPHA_Complex tmp, tmp2;
                    ALPHA_Complex val_c;
                    val_c.real = mat->values[ai].real;
                    val_c.imag = 0.0 - mat->values[ai].imag;

                    alpha_mul(tmp, alpha, mat->values[ai]);
                    alpha_mul(tmp2, tmp, x[index2(cc, ac, ldx)]);
                    alpha_add(y[index2(cc, cr, ldy)], y[index2(cc, cr, ldy)], tmp2);
                    alpha_mul(tmp, alpha, val_c);
                    alpha_mul(tmp2, tmp, x[index2(cc, cr, ldx)]);
                    alpha_add(y[index2(cc, ac, ldy)], y[index2(cc, ac, ldy)], tmp2);
                    // y[index2(cc, cr, ldy)] += alpha * mat->values[ai] * x[index2(cc, ac, ldx)];
                    // y[index2(cc, ac, ldy)] += alpha * mat->values[ai] * x[index2(cc, cr, ldx)];
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
