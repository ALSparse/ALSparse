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

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for(ALPHA_INT r = 0; r < m; ++r)
    {
		const ALPHA_INT row_start = A->pointers[r];
		const ALPHA_INT row_end = A->pointers[r + 1];
		ALPHA_INT row_indx = 1;
		for(ALPHA_INT i = row_start; i < row_end; i++)
		{
			ALPHA_INT row_eles = row_end - row_start;
			const ALPHA_Number v = A->values[i];

            ALPHA_INT c = r - row_eles + row_indx;
            if(i == row_end - 1)
			{
				alpha_madde(y[r], alpha, x[c]);
			}
            else
			{
				ALPHA_Number t;
				alpha_mul(t, alpha, A->values[i]);
				alpha_madde(y[r], t, x[c]);
			}
                
            row_indx ++;
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
