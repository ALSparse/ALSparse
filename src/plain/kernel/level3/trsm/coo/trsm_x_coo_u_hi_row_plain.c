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
                    {alpha_madde(temp, A->values[cr], y[col * ldy + out_y_col]);}
                    // temp += A->values[cr] * y[col * ldy + out_y_col];
            }
            ALPHA_Number t;
            alpha_mul(t, alpha, x[r * ldx + out_y_col]);
            alpha_sub(y[r * ldy + out_y_col], t, temp);
            // y[r * ldy + out_y_col] = (alpha * x[r * ldx + out_y_col] - temp);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
