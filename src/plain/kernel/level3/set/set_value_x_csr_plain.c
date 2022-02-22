#include "alphasparse/kernel_plain.h"

alphasparse_status_t
ONAME(ALPHA_SPMAT_CSR *A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const ALPHA_Number value)
{
	bool find = false;
	for(ALPHA_INT ai = A->rows_start[row]; ai < A->rows_end[row]; ++ai)
	{
		const ALPHA_INT ac = A->col_indx[ai];
		if(ac == col)
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
