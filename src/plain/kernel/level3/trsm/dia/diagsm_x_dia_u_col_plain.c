#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT c = 0; c < columns; ++c)
    {
        for (ALPHA_INT r = 0; r < A->rows; ++r)
        {
            alpha_mul(y[index2(c, r, ldy)], alpha, x[index2(c, r, ldx)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
