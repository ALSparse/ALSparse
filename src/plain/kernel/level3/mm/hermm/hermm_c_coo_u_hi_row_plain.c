#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
#ifdef PRINT
	printf("kernel hermm_c_coo_u_hi_row_plain called\n");
#endif
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    for (ALPHA_INT r = 0; r < m; ++r)
        for (ALPHA_INT c = 0; c < n; c++)
        {
            ALPHA_Complex t = {0, 0};
            alpha_mul(y[index2(r, c, ldy)], beta, y[index2(r, c, ldy)]);
            alpha_mul(t, alpha, x[index2(r, c, ldx)]);
            alpha_add(y[index2(r, c, ldy)], y[index2(r, c, ldy)], t);
            // y[index2(r, c, ldy)] = y[index2(r, c, ldy)] * beta + x[index2(r, c, ldx)] * alpha;
        }

    for (ALPHA_INT ai = 0; ai < mat->nnz; ai++)
    {
        ALPHA_INT ac = mat->col_indx[ai];
        ALPHA_INT r = mat->row_indx[ai];
        if (ac > r)
        {
            ALPHA_Complex origin_val = mat->values[ai];
            ALPHA_Complex conj_val = {mat->values[ai].real,-mat->values[ai].imag};

            // ALPHA_Complex t = alpha * mat->values[ai];
            ALPHA_Complex t,t_conj;
            alpha_mul(t, alpha, origin_val);
            alpha_mul(t_conj, alpha, conj_val);
            for (ALPHA_INT c = 0; c < n; ++c)
                alpha_madde(y[index2(r, c, ldy)], t, x[index2(ac, c, ldx)]);
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
                alpha_madde(y[index2(ac, c, ldy)], t_conj, x[index2(r, c, ldx)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
