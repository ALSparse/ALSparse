#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            alpha_mul(y[index2(i, j, ldy)], y[index2(i, j, ldy)], beta);

    for (ALPHA_INT ac = 0; ac < mat->cols; ++ac)
    {
        ALPHA_INT start = mat->pointers[ac];
        ALPHA_INT end   = mat->pointers[ac + 1];
        ALPHA_INT idx = 1;
        ALPHA_INT eles_num = end - start;
        for (ALPHA_INT ai = start; ai < end; ai++)
        {
            ALPHA_INT r = ac - eles_num + idx;
            if (ac > r)
            {
                ALPHA_Complex val;
                ALPHA_Complex val_c;
                val_c.real = mat->values[ai].real;
                val_c.imag = 0.0 - mat->values[ai].imag;
                alpha_mul(val_c, alpha, val_c);
                alpha_mul(val, alpha, mat->values[ai]);
                for (ALPHA_INT c = 0; c < n; ++c)
                    alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
                for (ALPHA_INT c = 0; c < n; ++c)
                    alpha_madde(y[index2(ac, c, ldy)], val_c, x[index2(r, c, ldx)]);
                    // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
            }
            else if(ac == r)
            {
                ALPHA_Complex val;
                alpha_mul(val, alpha, mat->values[ai]);
                for (ALPHA_INT c = 0; c < n; ++c)
                    alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
            }
            idx ++;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
