/** 
 * @Author: Zjj
 * @Date: 2020-06-24 09:32:52
 * @LastEditors: Zjj
 * @LastEditTime: 2020-06-24 16:36:00
 */
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <stdio.h>


alphasparse_status_t
ONAME(const ALPHA_Complex alpha,
      const ALPHA_SPMAT_COO *A,
      const ALPHA_Complex *x,
      const ALPHA_Complex beta,
      ALPHA_Complex *y)
{
#ifdef PRINT
	printf("kernel hermv_c_coo_n_lo_plain called\n");
#endif
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
	const ALPHA_INT nnz = A->nnz;
	

    for (ALPHA_INT i = 0; i < m; i++)
	{
		alpha_mul(y[i], y[i], beta);
	}

    for(ALPHA_INT i = 0; i < nnz; ++i)
	{
		const ALPHA_INT r = A->row_indx[i];
		const ALPHA_INT c = A->col_indx[i];
        const ALPHA_Complex origin_val = A->values[i];
        const ALPHA_Complex conj_val = {origin_val.real, - origin_val.imag};
		if(r < c)
		{
			continue;
		}
		ALPHA_Complex v,v_c;
		alpha_mul(v, origin_val, alpha);
		alpha_mul(v_c, conj_val, alpha);
		if(r == c)
		{
			alpha_madde(y[r], v, x[c]);
		}
		else
		{
			alpha_madde(y[r], v, x[c]);
			alpha_madde(y[c], v_c, x[r]);
	 	}
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;

}