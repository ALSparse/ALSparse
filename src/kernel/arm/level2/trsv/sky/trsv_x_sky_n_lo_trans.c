#include "alphasparse/kernel.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    alphasparse_status_t status = trsv_sky_n_hi(alpha, A, x, y);
    return status;
}
