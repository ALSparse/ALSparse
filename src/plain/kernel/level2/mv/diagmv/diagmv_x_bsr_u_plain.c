#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		             const ALPHA_SPMAT_BSR *A,
		             const ALPHA_Number *x,
		             const ALPHA_Number beta,
		             ALPHA_Number *y)
{
	const ALPHA_INT m = A->rows;
	 ALPHA_Number temp_1;
	alpha_setzero(temp_1);
	 ALPHA_Number temp_2;
	alpha_setzero(temp_2);
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		for(ALPHA_INT i = 0; i < m; ++i)
		{
			alpha_mul(temp_1, alpha, x[i]);
			alpha_mul(temp_2, beta, y[i]);
			alpha_add(y[i], temp_1, temp_2);  
			//y[i] = beta * y[i] + alpha * x[i];
		} 
	}else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for(ALPHA_INT i = 0; i < m; ++i)
		{
			alpha_mul(temp_1, alpha, x[i]);
			alpha_mul(temp_2, beta, y[i]);
			alpha_add(y[i], temp_1, temp_2);
			//y[i] = beta * y[i] + alpha * x[i];
		} 
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
