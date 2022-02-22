#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    ALPHA_Number diag[m];
    memset(diag, '\0', m * sizeof(ALPHA_Number));
    for (ALPHA_INT r = 0; r < m; r++)
    {
        const ALPHA_INT indx = A->pointers[r + 1] - 1;
        diag[r] = A->values[indx];
    }

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT c = A->cols - 1; c >= 0; c--)
        {
            ALPHA_Number temp;
            alpha_setzero(temp);
            for (ALPHA_INT ic = A->cols - 1; ic > c; ic--)
            {
                ALPHA_INT start = A->pointers[ic];
                ALPHA_INT end   = A->pointers[ic + 1];
                ALPHA_INT eles_num = ic - c;
                if(end - eles_num - 1 >= start)
                    alpha_madde(temp, A->values[end - eles_num - 1], y[out_y_col * ldy + ic]);
            }

            ALPHA_Number t;
            alpha_mul(t, alpha, x[out_y_col * ldx + c]);
            alpha_sub(t, t, temp);
            alpha_div(y[out_y_col * ldy + c], t, diag[c]);
            // y[out_y_col * ldy + c] = (alpha * x[out_y_col * ldx + c] - temp) / diag[c];
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
