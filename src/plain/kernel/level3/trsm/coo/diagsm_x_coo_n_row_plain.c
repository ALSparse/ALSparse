#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_Number diag[A->rows];

    memset(diag, '\0', A->rows * sizeof(ALPHA_Number));

    for (ALPHA_INT r = 0; r < A->nnz; r++)
    {
        if(A->row_indx[r] == A->col_indx[r])
        {
            // diag[A->row_indx[r]].real = A->values[r].real;
            // diag[A->row_indx[r]].imag = A->values[r].imag;
            diag[A->row_indx[r]] = A->values[r];
        }
    }
    
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        for (ALPHA_INT c = 0; c < columns; ++c)
        {
            ALPHA_Number t;
            alpha_mul(t, alpha, x[index2(r, c, ldx)]);
            alpha_div(y[index2(r, c, ldy)], t, diag[r]);
            // y[index2(r, c, ldy)] = alpha * x[index2(r, c, ldx)] / diag[r];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
