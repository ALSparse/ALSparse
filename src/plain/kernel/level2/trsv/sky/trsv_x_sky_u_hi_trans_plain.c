#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    alphasparse_status_t status = trsv_sky_u_lo_plain(alpha, A, x, y);
    return status;
}
