#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowA = mat->rows;

    ALPHA_Number diag[rowA];
    for (ALPHA_INT ar = 0; ar < rowA; ++ar)
    {   
        alpha_setzero(diag[ar]);
        ALPHA_INT idx = mat->pointers[ar + 1] - 1;
        diag[ar] = mat->values[idx];
   }

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            ALPHA_Number t;
            alpha_mul(t, alpha, diag[cr]);
            alpha_mul(t, t, x[index2(cc, cr, ldx)]);
            alpha_mul(y[index2(cc, cr, ldy)], beta, y[index2(cc, cr, ldy)]);
            alpha_add(y[index2(cc, cr, ldy)], y[index2(cc, cr, ldy)], t);
            // y[index2(cc, cr, ldy)] = beta * y[index2(cc, cr, ldy)] + alpha * diag[cr] * x[index2(cc, cr, ldx)];
        }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
