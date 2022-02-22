#include <string.h>

#include "alphasparse/opt.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <stdio.h>

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
      const ALPHA_SPMAT_COO *A,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    ALPHA_SPMAT_COO *transposed_mat;
    transpose_coo(A, &transposed_mat);
    alphasparse_status_t status = hermv_coo_n_hi(alpha,
                                                transposed_mat,
                                                x,
                                                beta,
                                                y);
    destroy_coo(transposed_mat);
    return status;
}