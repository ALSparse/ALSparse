#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t
diagmv_x_csr_n_omp(const ALPHA_Number alpha,
	               const ALPHA_SPMAT_CSR *A,
	               const ALPHA_Number *x,
	               const ALPHA_Number beta,
	               ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT thread_num = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
    for (ALPHA_INT i = 0; i < m; ++i)
    {
        register ALPHA_Number tmp;
        alpha_setzero(tmp);
        for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
        {
            if (A->col_indx[ai] == i)
            {
                alpha_mul(tmp, alpha, A->values[ai]);
                alpha_mule(tmp, x[i]);
                break;
            }
        }
        alpha_madd(y[i], beta, y[i], tmp);
        // y[i] = beta * y[i] + tmp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
	  const ALPHA_SPMAT_CSR *A,
	  const ALPHA_Number *x,
	  const ALPHA_Number beta,
	  ALPHA_Number *y)
{
    return diagmv_x_csr_n_omp(alpha, A, x, beta, y);
}
