#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t
ONAME(ALPHA_SPMAT_CSC *A, 
	  const ALPHA_INT row, 
	  const ALPHA_INT col,
	  const ALPHA_Number value)
{
	ALPHA_INT num_thread = alpha_get_thread_num();
	ALPHA_INT find = 0;

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread) reduction(+:find)
#endif
	for(ALPHA_INT ai = A->cols_start[col]; ai < A->cols_end[col]; ++ai)
	{
		const ALPHA_INT ar = A->row_indx[ai];
		if(ar == row)
		{
			A->values[ai] = value;
			find ++;
			ai = A->cols_end[col];
		}
	}

	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
