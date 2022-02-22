#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            alpha_mul(y[index2(i, j, ldy)], y[index2(i, j, ldy)], beta);

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT ac = 0; ac < mat->cols; ++ac)
        {
            ALPHA_INT start = mat->pointers[ac];
            ALPHA_INT end   = mat->pointers[ac + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end; ++ai)
            {
                ALPHA_INT cr = ac - eles_num + idx;
                if (ac > cr)
                {
                    ALPHA_Number t;
                    alpha_mul(t, alpha, mat->values[ai]);
                    alpha_madde(y[index2(cr, cc, ldy)], t, x[index2(ac, cc, ldx)]);
                    // y[index2(cr, cc, ldy)] += alpha * mat->values[ai] * x[index2(ac, cc, ldx)];
                }
                else if(ac == cr)
                    alpha_madde(y[index2(cr, cc, ldy)], alpha, x[index2(ac, cc, ldx)]);
                    // y[index2(cr, cc, ldy)] += alpha * x[index2(ac, cc, ldx)];

                idx++;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
