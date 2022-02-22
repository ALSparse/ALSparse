#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t 
ONAME(const ALPHA_Number alpha,
                 const ALPHA_SPMAT_CSC *A, 
                 const ALPHA_Number *x, 
                 const ALPHA_Number beta, 
                 ALPHA_Number *y)
{
    const int m = A->rows;
    const int n = A->cols;
    for (int j = 0; j < m; ++j)
    {
        alpha_mul(y[j], y[j], beta); 
        // y[j] *= beta;
    }
    ALPHA_Number tmp;
    alpha_setzero(tmp);     
    for (ALPHA_INT i = 0; i < n; i++)
    {
        for (ALPHA_INT ai = A->cols_start[i]; ai < A->cols_end[i]; ai++)
        {
            alpha_mul(tmp, A->values[ai], x[i]); 
            alpha_mul(tmp, alpha, tmp); 
            alpha_add(y[A->row_indx[ai]], y[A->row_indx[ai]], tmp);
            // y[A->col_indx[ai]] += alpha * A->values[ai] * x[i];
        } 
	}
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
