#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		               const ALPHA_SPMAT_COO* A,
		               const ALPHA_Number* x,
		               const ALPHA_Number beta,
		               ALPHA_Number* y)
{
    ALPHA_INT m = A->cols;
	ALPHA_INT nnz = A->nnz;
	for (ALPHA_INT i = 0; i < m; i++)
	{
		alpha_mule(y[i], beta);
	}
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        ALPHA_INT r = A->row_indx[i];
		ALPHA_INT c = A->col_indx[i];
		ALPHA_Number v;
		alpha_mul(v, A->values[i], x[r]);
		alpha_madde(y[c], alpha, v);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
