#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_INT nz,
                          const ALPHA_Number *y,
                          ALPHA_Number *x,
                          const ALPHA_INT *indx)
{
    return gthr_plain(nz, y, x, indx);
}