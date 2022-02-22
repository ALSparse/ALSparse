#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowC = mat->rows;
    ALPHA_INT colC = columns;

    for (ALPHA_INT c = 0; c < colC; ++c)
    {
        for (ALPHA_INT r = 0; r < rowC; ++r)
        {
            //y[index2(c, r, ldy)] = beta * y[index2(c, r, ldy)] + alpha * x[index2(c, r, ldy)];
            ALPHA_Number t1, t2;
            alpha_mul(t1, beta, y[index2(c, r, ldy)]);
            alpha_mul(t2, alpha, x[index2(c, r, ldy)]);
            alpha_add(y[index2(c, r, ldy)], t1, t2);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
