#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_size;

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT c = 0; c < n; ++c)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;

                for (ALPHA_INT lr = 0; lr < ll; ++lr)
                { // choose a inner row
                    ALPHA_Number extra;
                    alpha_setzero(extra);
                    for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
                    { // choose a block
                        ALPHA_INT ac = mat->col_indx[ai] * ll;
                        ALPHA_Number *blk = &mat->values[ai * ll * ll];

                        if (br == mat->col_indx[ai])
                        { // tlos is a diag block
                            alpha_add(extra, x[index2(c, r + lr, ldx)], extra);
                            for (ALPHA_INT lc = 0; lc < lr; ++lc)
                            {
                                alpha_madde(extra, blk[index2(lr, lc, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                        else if (br > mat->col_indx[ai])
                        {
                            for (ALPHA_INT lc = 0; lc < ll; ++lc)
                            {
                                alpha_madde(extra, blk[index2(lr, lc, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                    }
                    alpha_mul(y[index2(c, r + lr, ldy)], beta, y[index2(c, r + lr, ldy)]);
                    alpha_madde(y[index2(c, r + lr, ldy)], alpha, extra);
                }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT c = 0; c < n; ++c)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;

                for (ALPHA_INT lr = 0; lr < ll; ++lr)
                { // choose a inner row
                    ALPHA_Number extra;
                    alpha_setzero(extra);
                    for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
                    { // choose a block
                        ALPHA_INT ac = mat->col_indx[ai] * ll;
                        ALPHA_Number *blk = &mat->values[ai * ll * ll];

                        if (br == mat->col_indx[ai])
                        { // tlos is a diag block
                            alpha_add(extra, x[index2(c, r + lr, ldx)], extra);
                            for (ALPHA_INT lc = 0; lc < lr; ++lc)
                            {
                                alpha_madde(extra, blk[index2(lc, lr, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                        else if (br > mat->col_indx[ai])
                        {
                            for (ALPHA_INT lc = 0; lc < ll; ++lc)
                            {
                                alpha_madde(extra, blk[index2(lc, lr, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                    }
                    alpha_mul(y[index2(c, r + lr, ldy)], beta, y[index2(c, r + lr, ldy)]);
                    alpha_madde(y[index2(c, r + lr, ldy)], alpha, extra);
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
