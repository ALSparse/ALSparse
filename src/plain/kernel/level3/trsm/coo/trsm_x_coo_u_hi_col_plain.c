#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {        
        for (ALPHA_INT r = m - 1; r >= 0; r--)
        {
            ALPHA_Number temp; // = {.real = 0.0f, .imag = 0.0f};
            alpha_setzero(temp);
            for (ALPHA_INT cr = A->nnz - 1; cr >= 0; cr--)
            {
                int row = A->row_indx[cr];
                int col = A->col_indx[cr];
                if(row == r && col > r)
                    {alpha_madde(temp, A->values[cr], y[out_y_col * ldy + col]);}
                    // temp += A->values[cr] * y[out_y_col * ldy + col];
            }
            ALPHA_Number t;
            alpha_mul(t, alpha, x[out_y_col * ldx + r]);
            alpha_sub(y[out_y_col * ldy + r], t, temp);
            // y[out_y_col * ldy + r] = (alpha * x[out_y_col * ldx + r] - temp);
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
