#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

/*
* 
* Gather the elements of a full storage vector into a compressed vector by index
*
* details:
* x[i] = y[indx[i]], for i=0,1,... ,nz-1
*
* x         Sparse vector in compressed format
* y         full storage vector
* indx      The element index of x, stored in an array, with a length of at least nz
*
* input:
* x         Stored in an array, length is at least max(indx[i])
* y         Stored as an array, length is at least nz
* nz        Number of elements in vectors x and indx
* indx      The element index of x, stored in an array, with a length of at least nz
* 
* output:
* x         Stored in an array, length is at least max(indx[i])
*
*/

openspblas_sparse_status_t ONAME(const OPENSPBLAS_INT nz,
                          const OPENSPBLAS_Number *y,
                          OPENSPBLAS_Number *x,
                          const OPENSPBLAS_INT *indx)
{
    return gthr(nz, y, x, indx);
}