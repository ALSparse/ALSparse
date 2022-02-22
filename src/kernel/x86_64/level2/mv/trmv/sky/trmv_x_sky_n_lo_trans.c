#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t ONAME_omp(const ALPHA_Number alpha,
		                     const ALPHA_SPMAT_SKY *A,
		                     const ALPHA_Number *x,
		                     const ALPHA_Number beta,
		                     ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	const ALPHA_INT thread_num = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
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
            col_indx ++;
        }
    }
 
	return ALPHA_SPARSE_STATUS_SUCCESS;
}


alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_SKY *A,
		              const ALPHA_Number *x,
		              const ALPHA_Number beta,
		              ALPHA_Number *y)
{
    return ONAME_omp(alpha, A, x, beta, y);
}
