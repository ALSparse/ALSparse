#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t
ONAME(ALPHA_SPMAT_BSR *A, 
	  const ALPHA_INT nvalues, 
	  const ALPHA_INT *indx, 
	  const ALPHA_INT *indy, 
	  ALPHA_Number *values)
{
	ALPHA_INT num_thread = alpha_get_thread_num();
	ALPHA_INT find = 0;

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread) reduction(+:find)
#endif
	for(ALPHA_INT i = 0; i < nvalues; i++)
	{
		ALPHA_INT row = indx[i];
		ALPHA_INT col = indy[i];
		ALPHA_INT bs = A->block_size;
		ALPHA_INT block_row = row / bs;
		ALPHA_INT block_col = col / bs;
		ALPHA_INT block_row_inside = row % bs;
		ALPHA_INT block_col_inside = col % bs;
		for(ALPHA_INT ai = A->rows_start[block_row]; ai < A->rows_end[block_row]; ai++)
		{
			const ALPHA_INT ac = A->col_indx[ai];
			if(ac == block_col)
			{
				ALPHA_INT idx = 0;
				if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
					idx = ai * bs * bs + block_row_inside * bs + block_col_inside;
				else
					idx = ai * bs * bs + block_row_inside + block_col_inside * bs;
				
				A->values[idx] = values[i];
				find ++;
				break;
			}
		}
	}
	if(find)
		return ALPHA_SPARSE_STATUS_SUCCESS;
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;	
}
