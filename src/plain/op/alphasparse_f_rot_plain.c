#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_INT nz,
                          ALPHA_Number *x,
                          const ALPHA_INT *indx,
                          ALPHA_Number *y,
                          const ALPHA_Number c,
                          const ALPHA_Number s)
{
    return rot_plain(nz, x, indx, y, c, s);
}
