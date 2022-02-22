#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(ALPHA_SPMAT_CSC *A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const ALPHA_Number value)
{
	bool find = false;
	for(ALPHA_INT ai = A->cols_start[col]; ai < A->cols_end[col]; ++ai)
	{
		const ALPHA_INT ar = A->row_indx[ai];
		if(ar == row)
		{
			A->values[ai] = value;
			find = true;
			break;
		}
	}

	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
