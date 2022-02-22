#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

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

alphasparse_status_t ONAME(const ALPHA_INT nz,
                          ALPHA_Number *x,
                          const ALPHA_INT *indx,
                          ALPHA_Number *y,
                          const ALPHA_Number c,
                          const ALPHA_Number s)
{
    return rot(nz, x, indx, y, c, s);
}
