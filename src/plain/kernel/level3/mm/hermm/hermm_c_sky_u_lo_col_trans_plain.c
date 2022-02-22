#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
        {
            ALPHA_Complex t;
            alpha_mul(y[i + j * ldy], beta, y[i + j * ldy]);
            alpha_mul(t, alpha, x[i + j * ldx]);
            alpha_add(y[i + j * ldy], y[i + j * ldy], t);
            // y[i + j * ldy] = beta * y[i + j * ldy] + alpha * x[i + j * ldx];
        }  
            
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            ALPHA_INT start = mat->pointers[cr];
            ALPHA_INT end   = mat->pointers[cr + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end; ++ai)
            {
                ALPHA_INT ac = cr - eles_num + idx;
                if (ac < cr)
                {
                    ALPHA_Complex tmp;
                    ALPHA_Complex tmp_c;
                    tmp_c.real = mat->values[ai].real;
                    tmp_c.imag = 0.0 - mat->values[ai].imag;
                    alpha_mul(tmp_c, alpha, tmp_c);
                    alpha_mul(tmp, alpha, mat->values[ai]);
                    alpha_madde(y[index2(cc, cr, ldy)], tmp_c, x[index2(cc, ac, ldx)]);
                    alpha_madde(y[index2(cc, ac, ldy)], tmp, x[index2(cc, cr, ldx)]);
                    // y[index2(cc, cr, ldy)] += alpha * mat->values[ai] * x[index2(cc, ac, ldx)];
                    // y[index2(cc, ac, ldy)] += alpha * mat->values[ai] * x[index2(cc, cr, ldx)];
                }
                idx++;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
