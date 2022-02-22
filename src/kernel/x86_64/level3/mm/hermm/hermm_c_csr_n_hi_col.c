#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            alpha_mul(y[i + j * ldy], y[i + j * ldy], beta);
        
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            for (ALPHA_INT ai = mat->rows_start[cr]; ai < mat->rows_end[cr]; ++ai)
            {
                ALPHA_INT ac = mat->col_indx[ai];
                if (ac > cr)
                {
                    ALPHA_Complex tmp, tmp2;
                    ALPHA_Complex congval;
                    congval.real = mat->values[ai].real;
                    congval.imag = 0.0 - mat->values[ai].imag;

                    alpha_mul(tmp, alpha, mat->values[ai]);
                    alpha_mul(tmp2, tmp, x[index2(cc, ac, ldx)]);
                    alpha_add(y[index2(cc, cr, ldy)], y[index2(cc, cr, ldy)], tmp2);
                    
                    alpha_mul(tmp, alpha, congval);
                    alpha_mul(tmp2, tmp, x[index2(cc, cr, ldx)]);
                    alpha_add(y[index2(cc, ac, ldy)], y[index2(cc, ac, ldy)], tmp2);
                }
                else if (ac == cr)
                {
                    ALPHA_Complex tmp;
                    alpha_mul(tmp, alpha, mat->values[ai]);
                    alpha_mul(tmp, tmp, x[index2(cc, ac, ldx)]);
                    alpha_add(y[index2(cc, cr, ldy)], y[index2(cc, cr, ldy)], tmp);
                }
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
