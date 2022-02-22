#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_size;

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT r = 0; r < m; r += ll)
        {
            ALPHA_INT br = r / ll;
            for (ALPHA_INT lr = 0; lr < ll; ++lr)
                for (ALPHA_INT c = 0; c < n; c++)
                    // y[index2(r + lr, c, ldy)] *= beta;
                    alpha_mul(y[index2(r + lr, c, ldy)], beta, y[index2(r + lr, c, ldy)]);

            for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
            {
                ALPHA_INT lr, lc;
                ALPHA_INT ac = mat->col_indx[ai] * ll;
                ALPHA_Number *blk = &mat->values[ai * ll * ll];

                if (br == mat->col_indx[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                        for (lc = lr; lc < ll; ++lc)
                        {
                            ALPHA_Number val = blk[index2(lr, lc, ll)];
                            const ALPHA_Number *X = &x[index2(ac + lc, 0, ldx)];
                            alpha_mul(val, alpha, val);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                alpha_madde(y[index2(r+lr, c, ldy)], val, X[c]);
                        }
                }
                else if (br < mat->col_indx[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                        for (lc = 0; lc < ll; ++lc)
                        {
                            ALPHA_Number val = blk[index2(lr, lc, ll)];
                            const ALPHA_Number *X = &x[index2(ac + lc, 0, ldx)];
                            alpha_mul(val, alpha, val);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                alpha_madde(y[index2(r+lr, c, ldy)], val, X[c]);
                        }
                }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT r = 0; r < m; r += ll)
        {
            ALPHA_INT br = r / ll;
            for (ALPHA_INT lr = 0; lr < ll; ++lr)
                for (ALPHA_INT c = 0; c < n; c++)
                    // y[index2(r + lr, c, ldy)] *= beta;
                    alpha_mul(y[index2(r + lr, c, ldy)], beta, y[index2(r + lr, c, ldy)]);

            for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
            {
                ALPHA_INT lr, lc;
                ALPHA_INT ac = mat->col_indx[ai] * ll;
                ALPHA_Number *blk = &mat->values[ai * ll * ll];

                if (br == mat->col_indx[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                        for (lc = lr; lc < ll; ++lc)
                        {
                            ALPHA_Number val = blk[index2(lc, lr, ll)];
                            const ALPHA_Number *X = &x[index2(ac + lc, 0, ldx)];
                            alpha_mul(val, alpha, val);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                alpha_madde(y[index2(r+lr, c, ldy)], val, X[c]);
                        }
                }
                else if (br < mat->col_indx[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                        for (lc = 0; lc < ll; ++lc)
                        {
                            ALPHA_Number val = blk[index2(lc, lr, ll)];
                            const ALPHA_Number *X = &x[index2(ac + lc, 0, ldx)];
                            alpha_mul(val, alpha, val);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                alpha_madde(y[index2(r+lr, c, ldy)], val, X[c]);
                        }
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
