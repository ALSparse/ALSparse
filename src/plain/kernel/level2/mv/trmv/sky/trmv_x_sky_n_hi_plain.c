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

    for(ALPHA_INT c = 0; c < n; ++c)
    {
        const ALPHA_INT col_start = A->pointers[c];
		const ALPHA_INT col_end = A->pointers[c + 1];
        ALPHA_INT col_indx = 1;

        for(ALPHA_INT ai = col_start; ai < col_end; ++ai)
        {
            ALPHA_INT col_eles = col_end - col_start;
            ALPHA_INT r = c - col_eles + col_indx;
            ALPHA_Number t;
            alpha_mul(t, alpha, A->values[ai]);
            alpha_madde(y[r], t, x[c]);
			// y[r] += alpha* A->values[ai] * x[c];
            col_indx ++;
        }
    }

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
