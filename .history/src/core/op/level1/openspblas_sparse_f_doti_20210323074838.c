#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

OPENSPBLAS_Number ONAME(const OPENSPBLAS_INT nz,
                 const OPENSPBLAS_Number *x,
                 const OPENSPBLAS_INT *indx,
                 const OPENSPBLAS_Number *y)
{
    return doti(nz, x, indx, y);
}