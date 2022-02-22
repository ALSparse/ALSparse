#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_size;

    for (ALPHA_INT j = 0; j < n; ++j)
        for (ALPHA_INT i = 0; i < m; ++i)
        {
            alpha_mul(y[index2(j, i, ldy)], beta, y[index2(j, i, ldy)]);
        }

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT c = 0; c < n; c += ll)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;
                for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
                { // choose a block
                    ALPHA_Number *blk = &mat->values[ai * ll * ll];
                    for (ALPHA_INT cc = 0; cc < ll; ++cc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    { // choose a inner row

                        ALPHA_INT ac = mat->col_indx[ai] * ll;
                        ALPHA_Number extra;
                        alpha_setzero(extra);

                        for (ALPHA_INT lc = 0; lc < ll; ++lc)
                        {
                            alpha_madde(extra, blk[index2(lr, lc, ll)], x[index2(c + cc, ac + lc, ldx)]);
                        }
                        alpha_madde(y[index2(c + cc, r + lr, ldy)], alpha, extra);
                    }
                }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT c = 0; c < n; c += ll)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;
                for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
                { // choose a block
                    for (ALPHA_INT cc = 0; cc < ll; ++cc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    { // choose a inner row

                        ALPHA_INT ac = mat->col_indx[ai] * ll;
                        ALPHA_Number *blk = &mat->values[ai * ll * ll];
                        ALPHA_Number extra;
                        alpha_setzero(extra);

                        for (ALPHA_INT lc = 0; lc < ll; ++lc)
                        {
                            alpha_madde(extra, blk[index2(lc, lr, ll)], x[index2(c + cc, ac + lc, ldx)]);
                        }
                        alpha_madde(y[index2(c + cc, r + lr, ldy)], alpha, extra);
                    }
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
