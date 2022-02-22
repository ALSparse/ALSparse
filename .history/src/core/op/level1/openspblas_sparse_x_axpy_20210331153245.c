#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

/*
* 
* Add multiple scalar values ​​of a compressed vector to the full storage vector
*
* details:
* y := a*x + y
*
* a         a scalor value
* x         Sparse vector in compressed format
* y         full storage vector
*
* input:
* a         a scalor value
* x         Sparse vector in compressed format
* y         a full storage vector
* nz        Number of elements in vectors x and indx
* indx      The element index of x, stored in an array, with a length of at least nz
* 
* output:
* y         a full storage vector
*
*/

openspblas_sparse_status_t ONAME(const OPENSPBLAS_INT nz,
                          const OPENSPBLAS_Number a,
                          const OPENSPBLAS_Number *x,
                          const OPENSPBLAS_INT *indx,
                          OPENSPBLAS_Number *y)
{
    return axpy(nz, a, x, indx, y);
}