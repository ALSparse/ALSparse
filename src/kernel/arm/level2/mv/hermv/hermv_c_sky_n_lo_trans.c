#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Complex alpha,
	  const ALPHA_SPMAT_SKY *A,
	  const ALPHA_Complex *x,
	  const ALPHA_Complex beta,
	  ALPHA_Complex *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
    

	for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mul(y[i], y[i], beta);
	}
    for(ALPHA_INT r = 0; r < m; ++r)
    {
		const ALPHA_INT row_start = A->pointers[r];
		const ALPHA_INT row_end = A->pointers[r + 1];
		ALPHA_INT row_indx = 1;
		for(ALPHA_INT i = row_start; i < row_end; i++)
		{
			ALPHA_INT row_eles = row_end - row_start;
			ALPHA_Complex v;
			ALPHA_Complex v_c;
			v_c.real = A->values[i].real;
			v_c.imag = 0.0 - A->values[i].imag;
			alpha_mul(v_c, v_c, alpha);
			v.real = A->values[i].real;
			v.imag = A->values[i].imag;
			alpha_mul(v, v, alpha);
			if(i == row_end - 1)
			{
				ALPHA_INT c = r;
				alpha_madde(y[r], v_c, x[c]);
			}
			else
			{
				ALPHA_INT c = r - row_eles + row_indx;
				alpha_madde(y[r], v_c, x[c]);
				alpha_madde(y[c], v, x[r]);
				row_indx ++;
			}
		}
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
