#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

openspblas_sparse_status_t ONAME(const OPENSPBLAS_INT nz,
                          const OPENSPBLAS_Number *y,
                          OPENSPBLAS_Number *x,
                          const OPENSPBLAS_INT *indx)
{
    return gthr(nz, y, x, indx);
}