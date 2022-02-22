#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
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
    const ALPHA_INT thread_num = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for(ALPHA_INT i = 0; i < m; ++i)
	{
        alpha_mul(y[i], beta, y[i]);
		// y[i] *= beta;
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for(ALPHA_INT i = 1; i < m + 1; ++i)
	{
		const ALPHA_INT indx = A->pointers[i] - 1;
        ALPHA_Number tmp;
        alpha_mul(tmp, A->values[indx], x[i - 1]);     
        alpha_madde(y[i - 1], alpha, tmp);  
		// y[i - 1] += alpha * v * x[i - 1];
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
