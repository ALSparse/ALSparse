#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "stdio.h"
#include <string.h>
static alphasparse_status_t
gemv_bsr_for_each_thread(const ALPHA_Number alpha,
						   const ALPHA_SPMAT_BSR *A,
						   const ALPHA_Number *x,
						   const ALPHA_Number beta,
						   ALPHA_Number *y,
						   ALPHA_INT lrs,
						   ALPHA_INT lre)
{
	ALPHA_INT bs = A->block_size;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
	ALPHA_INT task_rows = (lre - lrs) * bs;
	// For matC, block_layout is defaulted as row_major
	ALPHA_Number *tmp = alpha_malloc(sizeof(ALPHA_Number) * task_rows);
	memset(tmp, 0, sizeof(ALPHA_Number) * task_rows);
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		for (ALPHA_INT i = lrs, j = 0; i < lre; i++, j++)
		{
			for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++)
			{
				//TODO Code here if unroll is needed
				for (ALPHA_INT row_inner = 0; row_inner < bs; row_inner++)
				{
					for (ALPHA_INT col_inner = 0; col_inner < bs; col_inner++)
					{
						alpha_madde(tmp[bs * j + row_inner], A->values[ai * bs * bs + row_inner * bs + col_inner], x[bs * A->col_indx[ai] + col_inner]);
					}
					// over for block
				}
			}
		}
	}
	// For Fortran, block_layout is defaulted as col_major
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
		for (ALPHA_INT i = lrs, j = 0; i < lre; i++, j++)
		{
			for (ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ai++)
			{
				// block[ai]: [i][A->cols[ai]]
				for (ALPHA_INT col_inner = 0; col_inner < bs; col_inner++)
				{
					for (ALPHA_INT row_inner = 0; row_inner < bs; row_inner++)
					{
						alpha_madde(tmp[bs * j + row_inner], A->values[ai * bs * bs + col_inner * bs + row_inner], x[bs * A->col_indx[ai] + col_inner]);
					}
					// over for block
				}
			}
		}
	}
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	for (ALPHA_INT m = lrs * bs, m_t = 0; m < lre * bs; m++, m_t++)
	{
		alpha_mul(y[m], y[m], beta);
		alpha_madde(y[m], tmp[m_t], alpha);
	}
	free(tmp);
	return ALPHA_SPARSE_STATUS_SUCCESS;
}

static alphasparse_status_t
gemv_bsr_omp(const ALPHA_Number alpha,
			   const ALPHA_SPMAT_BSR *A,
			   const ALPHA_Number *x,
			   const ALPHA_Number beta,
			   ALPHA_Number *y)
{
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT thread_num = alpha_get_thread_num();

	ALPHA_INT partition[thread_num + 1];
	balanced_partition_row_by_nnz(A->rows_end, m_inner, thread_num, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
	{
		ALPHA_INT tid = alpha_get_thread_id();
		ALPHA_INT local_m_s = partition[tid];
		ALPHA_INT local_m_e = partition[tid + 1];
		gemv_bsr_for_each_thread(alpha, A, x, beta, y, local_m_s, local_m_e);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		   const ALPHA_SPMAT_BSR *A,
		   const ALPHA_Number *x,
		   const ALPHA_Number beta,
		   ALPHA_Number *y)
{
	return gemv_bsr_omp(alpha, A, x, beta, y);
}
