#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = mat->cols;
    for(ALPHA_INT j=0; j < m; ++j)
    for(ALPHA_INT i=0; i < columns; ++i)    
    {
        alpha_mul(y[index2(j,i,ldy)], beta, y[index2(j,i,ldy)]);//y[i] *= beta;
    }

    for(ALPHA_INT ac = 0; ac<n; ++ac)
    { 
        for(ALPHA_INT ai = mat->cols_start[ac]; ai < mat->cols_end[ac]; ++ai)
        {
            ALPHA_INT ar = mat->row_indx[ai];
            ALPHA_Number val;
            alpha_conj(val, mat->values[ai]);
            alpha_mule(val, alpha);

            if(ac > ar)
            { // val @ [ar, ac] & [ac, ar]
                for(ALPHA_INT cc = 0; cc < columns; ++cc)
                {
                    alpha_madde(y[index2(ar, cc, ldy)], val, x[index2(ac, cc, ldx)]);
                    alpha_madde(y[index2(ac, cc, ldy)], val, x[index2(ar, cc, ldx)]);
                }
            }
            else if(ac == ar)
            {
                for(ALPHA_INT cc = 0; cc < columns; ++cc)
                {
                    alpha_madde(y[index2(ar, cc, ldy)], val, x[index2(ac, cc, ldx)]);
                }
                
            }

        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}