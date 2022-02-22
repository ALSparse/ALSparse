#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static alphasparse_status_t
trmv_x_csr_u_lo_omp(const ALPHA_Number alpha,
                    const ALPHA_SPMAT_CSR *A,
                    const ALPHA_Number *x,
                    const ALPHA_Number beta,
                    ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(ALPHA_INT i = 0;i < m; ++i)
    {
        ALPHA_Number tmp = x[i];
        for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
        {
            const ALPHA_INT col = A->col_indx[ai];
            if(col < i)
            {
                alpha_madde(tmp, A->values[ai], x[col]);
            }
            else                                                                        
            {
                break;
            }
        }
        alpha_mule(tmp, alpha);
        alpha_mule(y[i], beta);
        alpha_adde(y[i], tmp);
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
    return trmv_x_csr_u_lo_omp(alpha, A, x, beta, y);
}
