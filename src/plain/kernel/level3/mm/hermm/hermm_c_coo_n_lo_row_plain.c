/** 
 * @Author: Zjj
 * @Date: 2020-06-25 11:21:20
 * @LastEditors: Zjj
 * @LastEditTime: 2020-06-25 14:44:00
 */
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
#ifdef PRINT
	printf("kernel hermm_c_coo_n_lo_row_plain called\n");
#endif
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;

    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            alpha_mul(y[i * ldy + j], y[i * ldy + j], beta);

    for (ALPHA_INT ai = 0; ai < mat->nnz; ai++)
    {
        ALPHA_INT ac = mat->col_indx[ai];
        ALPHA_INT r = mat->row_indx[ai];
        ALPHA_Complex origin_val = mat->values[ai];
        ALPHA_Complex conj_val = {origin_val.real,-origin_val.imag};
        ALPHA_Complex t,t_conj;
        if (ac < r)
        {
            alpha_mul(t, alpha, origin_val);
            alpha_mul(t_conj, alpha, conj_val);

            for (ALPHA_INT c = 0; c < n; ++c)
                alpha_madde(y[index2(r, c, ldy)], t, x[index2(ac, c, ldx)]);
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                alpha_madde(y[index2(ac, c, ldy)], t_conj, x[index2(r, c, ldx)]);
                // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
        }
        else if (ac == r)
        {
            // ALPHA_Complex val = alpha * mat->values[ai]; 
            ALPHA_Complex val;
            alpha_mul(val, alpha, origin_val);
            for (ALPHA_INT c = 0; c < n; ++c)
                alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
