#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r < m; r++)
        {
            ALPHA_Number temp;
            alpha_setzero(temp);
            for (ALPHA_INT ai = A->rows_start[r]; ai < A->rows_end[r]; ai++)
            {
                ALPHA_INT ac = A->col_indx[ai];
                if (ac < r)
                {
                    alpha_madde(temp, A->values[ai], y[out_y_col * ldy + ac]);
                }
            }
            ALPHA_Number t;
            alpha_setzero(t);
            alpha_mul(t, alpha, x[out_y_col * ldx + r]);
            alpha_sub(y[out_y_col * ldy + r], t, temp);
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
