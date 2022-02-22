#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(ALPHA_SPMAT_COO *A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const ALPHA_Number value)
{
	bool find = false;
	for(ALPHA_INT ai = 0; ai < A->nnz; ++ai)
		if(A->row_indx[ai] == row && A->col_indx[ai] == col)
		{
			A->values[ai] = value;
			find = true;
			break;
		}
	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
