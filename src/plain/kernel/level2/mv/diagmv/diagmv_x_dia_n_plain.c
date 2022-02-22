#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		             const ALPHA_SPMAT_DIA *A,
		             const ALPHA_Number *x,
		             const ALPHA_Number beta,
		             ALPHA_Number *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	const ALPHA_INT diags = A->ndiag;
	ALPHA_INT coll = -1;
	for(ALPHA_INT i = 0; i < diags; ++i)
	 {
		if(A->distance[i] == 0)
	 	{

			for(ALPHA_INT j = 0; j < m; ++j)
	 		{
				alpha_mul(y[j], beta, y[j]);
				ALPHA_Number v;
				alpha_mul(v, alpha, A->values[i * m + j]);
				alpha_madde(y[j], v, x[j]);
				if( !(alpha_iszero(A->values[i * m + j])) ){
					coll = j + 1;
					break;
				}
			}
			for(ALPHA_INT j = coll ;j < m ; j++){
				ALPHA_Number val = A->values[i * m + j];
				if(alpha_iszero(val)){
					continue;	
				}
				alpha_mul(y[j], beta, y[j]);
				ALPHA_Number v;
				alpha_mul(v, alpha, val);
				alpha_madde(y[j], v, x[j]);
			}
			break;
		}
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
