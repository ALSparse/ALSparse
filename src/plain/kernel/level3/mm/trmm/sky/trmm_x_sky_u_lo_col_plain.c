#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            alpha_mul(y[index2(j, i, ldy)], y[index2(j, i, ldy)], beta);

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            ALPHA_INT start = mat->pointers[cr];
            ALPHA_INT end   = mat->pointers[cr + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end; ++ai)
            {
                ALPHA_INT ac = cr - eles_num + idx;
                if (ac < cr)
                {
                    ALPHA_Number t;
                    alpha_mul(t, alpha, mat->values[ai]);
                    alpha_madde(y[index2(cc, cr, ldy)], t, x[index2(cc, ac, ldx)]);
                    // y[index2(cc, cr, ldy)] += alpha * mat->values[ai] * x[index2(cc, ac, ldx)];
                }
                else if(ac == cr)
                    alpha_madde(y[index2(cc, cr, ldy)], alpha, x[index2(cc, ac, ldx)]);
                    // y[index2(cc, cr, ldy)] += alpha * x[index2(cc, ac, ldx)];

                idx++;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
