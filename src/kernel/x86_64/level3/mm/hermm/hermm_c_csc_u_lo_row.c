#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = mat->cols;

    //prepare all y with beta
    for (ALPHA_INT j = 0; j < m; ++j)
    for (ALPHA_INT i = 0; i < columns; ++i)
    {
        ALPHA_Complex tmp1, tmp2;
        alpha_mul(tmp1, alpha, x[index2(j,i,ldx)]);
        alpha_mul(tmp2, beta, y[index2(j,i,ldy)]);
        alpha_add(y[index2(j,i,ldy)], tmp1, tmp2);
    }

    for(ALPHA_INT ac = 0; ac<n; ++ac)
    { 
        for(ALPHA_INT ai = mat->cols_start[ac]; ai < mat->cols_end[ac]; ++ai)
        {
            ALPHA_INT ar = mat->row_indx[ai];
            ALPHA_Complex val;
            ALPHA_Complex val_c;
            val_c.real = mat->values[ai].real;
            val_c.imag = 0.0 - mat->values[ai].imag;
            alpha_mul(val_c, alpha, val_c);
            alpha_mul(val, alpha, mat->values[ai]);

            if(ac < ar)
            {
                for(ALPHA_INT cc = 0; cc < columns; ++cc)
                {
                    alpha_madde(y[index2(ar, cc, ldy)], val, x[index2(ac, cc, ldx)]);
                    alpha_madde(y[index2(ac, cc, ldy)], val_c, x[index2(ar, cc, ldx)]);
                }
            }                
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
