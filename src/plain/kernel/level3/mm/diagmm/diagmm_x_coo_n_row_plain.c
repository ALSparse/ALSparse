#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT n = columns;
    ALPHA_INT _nnz = mat->nnz;
    ALPHA_INT or = 0;

    for (ALPHA_INT nnz = 0; nnz < _nnz; ++nnz)
    {
        ALPHA_INT r = mat->row_indx[nnz];
        ALPHA_Number *Y = &y[index2(r, 0, ldy)];
        while(or <= r)
        {
            ALPHA_Number *TY = &y[index2(or, 0, ldy)];
            for (ALPHA_INT c = 0; c < n; c++)
                // Y[c] = Y[c] * beta;
                alpha_mule(TY[c], beta);

            or++;
        }

        if (mat->col_indx[nnz] == r)
        {
            ALPHA_Number val;
            alpha_mul(val, alpha, mat->values[nnz]);
            const ALPHA_Number *X = &x[index2(mat->col_indx[nnz], 0, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                alpha_madde(Y[c], val, X[c]);
                // Y[c] += val * X[c];
        }
    }

    while(or < mat->rows)
    {
        ALPHA_Number *TY = &y[index2(or, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
            // Y[c] = Y[c] * beta;
            alpha_mul(TY[c], TY[c], beta);

        or++;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
