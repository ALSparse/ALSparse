#include "openspblas/spapi.h"
#include "openspblas/kernel.h"
#include "openspblas/util.h"

openspblas_sparse_status_t ONAME(const OPENSPBLAS_INT nz,
                          OPENSPBLAS_Number *x,
                          const OPENSPBLAS_INT *indx,
                          OPENSPBLAS_Number *y,
                          const OPENSPBLAS_Number c,
                          const OPENSPBLAS_Number s)
{
    return rot(nz, x, indx, y, c, s);
}
