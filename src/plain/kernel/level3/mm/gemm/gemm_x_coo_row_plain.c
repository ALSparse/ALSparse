#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT n = columns;
    ALPHA_INT r = 0;//mat->row_indx[0];

    for (ALPHA_INT nn = 0; nn < mat->nnz; ++nn)
    {
        ALPHA_INT cr = mat->row_indx[nn];
        ALPHA_Number *Y = &y[index2(cr, 0, ldy)];
        while(cr >= r)
        {
            ALPHA_Number *TY = &y[index2(r, 0, ldy)];
            for (ALPHA_INT c = 0; c < n; c++)
                alpha_mul(TY[c], TY[c], beta);

            r++;
        }

        ALPHA_Number val;
        alpha_mul(val, alpha, mat->values[nn]);
        const ALPHA_Number *X = &x[index2(mat->col_indx[nn], 0, ldx)];
        for (ALPHA_INT c = 0; c < n; ++c)
            alpha_madde(Y[c], val, X[c]);
            // Y[c] += val * X[c];
    }

    while(mat->rows > r)
    {
        ALPHA_Number *TY = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
            alpha_mul(TY[c], TY[c], beta);

        r++;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
