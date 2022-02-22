
#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
      const ALPHA_SPMAT_CSR *A,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    for (ALPHA_INT j = 0; j < n; ++j)
    {
        alpha_mule(y[j], beta);
    }
    for (ALPHA_INT i = 0; i < m; i++)
    {
        for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++)
        {
            ALPHA_Number val;
            alpha_mul(val, alpha, A->values[ai]);
            alpha_madde(y[A->col_indx[ai]], val, x[i]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
