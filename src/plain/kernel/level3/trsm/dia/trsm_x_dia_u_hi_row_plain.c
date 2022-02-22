#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    ALPHA_INT main_diag_pos = 0;

    for (ALPHA_INT i = 0; i < A->ndiag; i++)
        if(A->distance[i] == 0)
        {
            main_diag_pos = i;
            break;
        }

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = m - 1; r >= 0; r--)
        {
            ALPHA_Number temp;
            alpha_setzero(temp);
            for (ALPHA_INT ndiag = main_diag_pos + 1; ndiag < A->ndiag; ndiag++)
            {
                if (m - A->distance[ndiag] > r)
                {
                    ALPHA_INT ac = r + A->distance[ndiag];
                    alpha_madde(temp, A->values[ndiag * A->lval + r], y[ac * ldy + out_y_col]);
                    // temp += A->values[ndiag * A->lval + r] * y[ac * ldy + out_y_col];
                }
            }
            ALPHA_Number t;
            alpha_setzero(t);
            alpha_mul(t, alpha, x[r * ldx + out_y_col]);
            alpha_sub(y[r * ldy + out_y_col], t, temp);
            // y[r * ldy + out_y_col] = alpha * x[r * ldx + out_y_col] - temp;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
