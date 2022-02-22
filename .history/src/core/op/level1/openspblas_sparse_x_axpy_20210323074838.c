#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

openspblas_sparse_status_t ONAME(const OPENSPBLAS_INT nz,
                          const OPENSPBLAS_Number a,
                          const OPENSPBLAS_Number *x,
                          const OPENSPBLAS_INT *indx,
                          OPENSPBLAS_Number *y)
{
    return axpy(nz, a, x, indx, y);
}