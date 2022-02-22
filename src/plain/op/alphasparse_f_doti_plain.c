#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"

ALPHA_Number ONAME(const ALPHA_INT nz,
                 const ALPHA_Number *x,
                 const ALPHA_INT *indx,
                 const ALPHA_Number *y)
{
    return doti_plain(nz, x, indx, y);
}