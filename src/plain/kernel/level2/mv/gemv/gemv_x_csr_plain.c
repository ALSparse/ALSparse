
#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
      const ALPHA_SPMAT_CSR *A,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    ALPHA_INT m = A->rows;
    for (ALPHA_INT r = 0; r < m; r++)
    {
        alpha_mule(y[r], beta);
        ALPHA_Number tmp;
        alpha_setzero(tmp);
        for (ALPHA_INT ai = A->rows_start[r]; ai < A->rows_end[r]; ai++)
        {
            alpha_madde(tmp, A->values[ai], x[A->col_indx[ai]]);
        }
        alpha_madde(y[r], alpha, tmp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
