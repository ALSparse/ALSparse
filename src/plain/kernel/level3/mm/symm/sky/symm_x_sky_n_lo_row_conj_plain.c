#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
#ifdef COMPLEX
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    for (ALPHA_INT r = 0; r < m; ++r)
    {
        for (ALPHA_INT c = 0; c < n; c++)
            alpha_mul(y[index2(r, c, ldy)], y[index2(r, c, ldy)], beta);
            // y[index2(r, c, ldy)] *= beta;
        
        ALPHA_INT start = mat->pointers[r];
        ALPHA_INT end   = mat->pointers[r + 1];
        ALPHA_INT idx = 1;
        ALPHA_INT eles_num = end - start;

        for (ALPHA_INT ai = start; ai < end; ai++)
        {
            ALPHA_INT ac = r - eles_num + idx;
            if (ac < r)
            {
                ALPHA_Number val;
                alpha_mul_3c(val, alpha, mat->values[ai]);
                for (ALPHA_INT c = 0; c < n; ++c)
                    alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
                for (ALPHA_INT c = 0; c < n; ++c)
                    alpha_madde(y[index2(ac, c, ldy)], val, x[index2(r, c, ldx)]);
                    // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
            }
            else if (ac == r)
            {
                ALPHA_Number val;
                alpha_mul_3c(val, alpha, mat->values[ai]);
                for (ALPHA_INT c = 0; c < n; ++c)
                    alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
            }
            idx ++;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
#else
	return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
}
