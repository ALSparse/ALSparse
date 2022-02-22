#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++) //由于unit，对角线元素都不用取了
    {
        //y[r] = alpha * x[r];
        alpha_mul(y[r], alpha, x[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
