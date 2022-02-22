#include "openspblas/spapi.h"
#include "openspblas/kernel.h"
#include "openspblas/util.h"

/*
* 
* Gather the elements of a full storage vector into the compressed vector format
* and zero the elements at the corresponding positions in the original vector
*
* details:
* x[i] = c*x[i] + s*y[indx[i]]
* y[indx[i]] = c*y[indx[i]]- s*x[i]
*
* x         Sparse vector in compressed format
* y         full storage vector
* indx      The element index of x, stored in an array, with a length of at least nz
* c         a scalar value
* s         a scalar value
*
* input:
* x         Stored in an array, length is at least max(indx[i])
* y         Stored as an array, length is at least nz
* nz        Number of elements in vectors x and indx
* indx      The element index of x, stored in an array, with a length of at least nz
* c         a scalar value
* s         a scalar value
* 
* output:
* x         Compressed vector x
* y         the updated y
*
*/

openspblas_sparse_status_t ONAME(const OPENSPBLAS_INT nz,
                          OPENSPBLAS_Number *x,
                          const OPENSPBLAS_INT *indx,
                          OPENSPBLAS_Number *y,
                          const OPENSPBLAS_Number c,
                          const OPENSPBLAS_Number s)
{
    return rot(nz, x, indx, y, c, s);
}
