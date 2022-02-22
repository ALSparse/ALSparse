#include "alphasparse/kernel.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "alphasparse/opt.h"
#include <string.h>
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
				const ALPHA_SPMAT_BSR *A,
				const ALPHA_Number *x,
				const ALPHA_Number beta,
				ALPHA_Number *y)
{
	const ALPHA_INT thread_num = alpha_get_thread_num();
	const ALPHA_INT m = A->rows * A->block_size;
	const ALPHA_INT n = A->cols * A->block_size;
	const ALPHA_INT bs = A->block_size;
	const ALPHA_INT bs2 = bs * bs;
	// assert(m==n);
	ALPHA_INT b_rows = A->rows;
	ALPHA_INT b_cols = A->cols;
	if (b_rows != b_cols)
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for (ALPHA_INT j = 0; j < A->rows * A->block_size; j++)
	{
		alpha_mul(y[j], y[j], beta);
	}
	ALPHA_INT partition[thread_num + 1];
	balanced_partition_row_by_nnz(A->rows_end, b_rows, thread_num, partition);
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
		{
			ALPHA_INT tid = alpha_get_thread_id();
			for (ALPHA_INT br = partition[tid]; br < partition[tid + 1]; br++)
			{
				ALPHA_INT row = br * bs;
				ALPHA_INT block_start = A->rows_start[br], block_end = A->rows_end[br];
				ALPHA_INT upper_start = alpha_lower_bound(&A->col_indx[block_start], &A->col_indx[block_end], br) - A->col_indx;
				for (ALPHA_INT ai = upper_start; ai < block_end; ai++)
				{
					ALPHA_INT bc = A->col_indx[ai];
					ALPHA_INT col = bc * bs;
					ALPHA_INT a0_idx = ai * bs2;
					ALPHA_Number val_orig;
					ALPHA_Number temp_orig;
					// diagonal block containing diagonal entry
					if (bc == br)
					{
						for (ALPHA_INT b_row = 0; b_row < bs; b_row++)
						{
							for (ALPHA_INT b_col = b_row; b_col < bs; b_col++)
							{
								alpha_mul(temp_orig, alpha, A->values[a0_idx + b_row * bs + b_col]);
								alpha_madde(y[b_row + row], temp_orig, x[col + b_col]);
							}
						}
					}
					else
					{
						for (ALPHA_INT b_row = 0; b_row < bs; b_row++)
						{
							ALPHA_INT b_col = 0;
							for (; b_col < bs; b_col++)
							{
								alpha_mul(temp_orig, alpha, A->values[a0_idx + b_row * bs + b_col]);
								alpha_madde(y[b_row + row], temp_orig, x[col + b_col]);
							}
						}
					}
				}
			}
		}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
		{
			ALPHA_INT tid = alpha_get_thread_id();
			for (ALPHA_INT br = partition[tid]; br < partition[tid + 1]; br++)
			{
				ALPHA_INT row = br * bs;
				ALPHA_INT block_start = A->rows_start[br], block_end = A->rows_end[br];
				ALPHA_INT upper_start = alpha_lower_bound(&A->col_indx[block_start], &A->col_indx[block_end], br) - A->col_indx;

				for (ALPHA_INT ai = upper_start; ai < block_end; ++ai)
				{
					ALPHA_INT bc = A->col_indx[ai];
					ALPHA_INT col = bc * bs;
					ALPHA_INT a0_idx = ai * bs2;
					ALPHA_Number val_orig;
					ALPHA_Number temp_orig;
					// diagonal block containing diagonal entry
					if (bc == br)
					{
						for (ALPHA_INT b_col = 0; b_col < bs; b_col++)
						{
							for (ALPHA_INT b_row = 0; b_row <= b_col; b_row++)
							{
								alpha_mul(temp_orig, alpha, A->values[a0_idx + b_col * bs + b_row]);
								alpha_madde(y[b_row + row], temp_orig, x[col + b_col]);
							}
						}
					}
					else
					{
						for (ALPHA_INT b_col = 0; b_col < bs; b_col++)
						{
							for (ALPHA_INT b_row = 0; b_row < bs; b_row++)
							{
								alpha_mul(temp_orig, alpha, A->values[a0_idx + b_col * bs + b_row]);
								alpha_madde(y[b_row + row], temp_orig, x[col + b_col]);
							}
						}
					}
				}
			}
		}
	}
	else
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}