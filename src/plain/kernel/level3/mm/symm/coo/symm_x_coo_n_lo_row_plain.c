#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            alpha_mul(y[i * ldy + j], y[i * ldy + j], beta);
            // y[i] *= beta;


    for (ALPHA_INT ai = 0; ai < mat->nnz; ai++)
    {
        ALPHA_INT ac = mat->col_indx[ai];
        ALPHA_INT r = mat->row_indx[ai];
        if (ac < r)
        {
            ALPHA_Number val;
            alpha_mul(val, alpha, mat->values[ai]);
            for (ALPHA_INT c = 0; c < n; ++c)
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
                alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
            for (ALPHA_INT c = 0; c < n; ++c)
                alpha_madde(y[index2(ac, c, ldy)], val, x[index2(r, c, ldx)]);
                // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
        }
        else if (ac == r)
        {
            ALPHA_Number val;
            alpha_mul(val, alpha, mat->values[ai]);
            for (ALPHA_INT c = 0; c < n; ++c)
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
                alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
