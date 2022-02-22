#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
#ifdef COMPLEX
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_size;

    for (ALPHA_INT c = 0; c < n; c++)
        for (ALPHA_INT r = 0; r < m; ++r)
        {
            alpha_mul(y[index2(c, r, ldy)], beta, y[index2(c, r, ldy)]);
            alpha_madde(y[index2(c, r, ldy)], alpha, x[index2(c, r, ldx)]);
        }

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT matC = 0; matC < n; matC += ll)
        for (ALPHA_INT R = 0; R < m; R += ll)
        {
            ALPHA_INT br = R / ll;
            
            for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
            {
                ALPHA_INT ac = mat->col_indx[ai] * ll;
                ALPHA_Number *blk = &mat->values[ai*ll*ll];
                
                if(br == mat->col_indx[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        ALPHA_Number extra;
                        alpha_setzero(extra);//x[index2(matC+lc, R+lr, ldx)];
                        for (ALPHA_INT i=0; i < lr; ++i)
                        {
                            alpha_madde_2c(extra, blk[index2(lr, i, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        for (ALPHA_INT i=lr+1; i < ll; ++i)
                        {
                            alpha_madde_2c(extra, blk[index2(i, lr, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                    }
                }
                else if(br > mat->col_indx[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        ALPHA_Number extra,extra_sym;
                        alpha_setzero(extra);
                        alpha_setzero(extra_sym);
                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            alpha_madde_2c(extra, blk[index2(lr, i, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }

                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            alpha_madde_2c(extra_sym, blk[index2(i, lr, ll)], x[index2(matC+lc, R+i, ldx)]);
                        }                        
                        alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                        alpha_madde(y[index2(matC+lc, ac+lr, ldy)], alpha, extra_sym);
                    }
                }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT matC = 0; matC < n; matC += ll)
        for (ALPHA_INT R = 0; R < m; R += ll)
        {
            ALPHA_INT br = R / ll;
            
            for (ALPHA_INT ai = mat->rows_start[br]; ai < mat->rows_end[br]; ++ai)
            {
                ALPHA_INT ac = mat->col_indx[ai] * ll;
                ALPHA_Number *blk = &mat->values[ai*ll*ll];
                
                if(br == mat->col_indx[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        ALPHA_Number extra;
                        alpha_setzero(extra);//x[index2(matC+lc, R+lr, ldx)];
                        for (ALPHA_INT i=0; i < lr; ++i)
                        {
                            alpha_madde_2c(extra, blk[index2(i, lr, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        for (ALPHA_INT i=lr+1; i < ll; ++i)
                        {
                            alpha_madde_2c(extra, blk[index2(lr, i, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                    }
                }
                else if(br > mat->col_indx[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        ALPHA_Number extra,extra_sym;
                        alpha_setzero(extra);
                        alpha_setzero(extra_sym);
                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            alpha_madde_2c(extra, blk[index2(i, lr, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }

                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            alpha_madde_2c(extra_sym, blk[index2(lr, i, ll)], x[index2(matC+lc, R+i, ldx)]);
                        }                        
                        alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                        alpha_madde(y[index2(matC+lc, ac+lr, ldy)], alpha, extra_sym);
                    }
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
#else
	return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
}
