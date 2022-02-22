#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            alpha_mul(y[i + j * ldy], y[i + j * ldy], beta);

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT nn = 0; nn < mat->nnz; ++nn)
        {
            ALPHA_Number ctmp;
            alpha_mul(ctmp, mat->values[nn], x[index2(cc, mat->col_indx[nn], ldx)]); 
            alpha_madde(y[index2(cc, mat->row_indx[nn], ldy)], alpha, ctmp);
            // y[index2(mat->row_indx[nn], cc, ldy)] += alpha * mat->values[nn] * x[index2(mat->col_indx[nn], cc, ldx)];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
