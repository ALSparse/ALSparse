#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_SKY *A,
		              const ALPHA_Number *x,
		              const ALPHA_Number beta,
		              ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    for(ALPHA_INT i = 0; i < m; ++i)
	{
		alpha_mul(y[i], beta, y[i]);
	}
    for(ALPHA_INT r = 0; r < m; ++r)
    {
		const ALPHA_INT row_start = A->pointers[r];
		const ALPHA_INT row_end = A->pointers[r + 1];
		ALPHA_INT row_indx = 1;
		for(ALPHA_INT i = row_start; i < row_end; i++)
		{
			ALPHA_INT row_eles = row_end - row_start;
            ALPHA_INT c = r - row_eles + row_indx;
			ALPHA_Number t;
            alpha_mul(t, alpha, A->values[i]);
            alpha_madde(y[r], t, x[c]);
            // y[r] += alpha * v * x[c];
            row_indx ++;
		}
    }

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
