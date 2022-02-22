#include "alphasparse/kernel.h"
#ifdef _OPENMP
#include<omp.h>
#endif
#include"alphasparse/opt.h"
#include<string.h>
#include "stdio.h"
#include <stdlib.h>
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
	ALPHA_INT b_rows = A->rows;
	ALPHA_INT b_cols = A->cols;

	if (b_rows != b_cols)
		return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	
	ALPHA_INT partition[thread_num + 1];
	balanced_partition_row_by_nnz(A->rows_end, b_rows, thread_num, partition);
	ALPHA_Number **tmp = (ALPHA_Number **)malloc(sizeof(ALPHA_Number *) * thread_num);

#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
	{
		const ALPHA_INT tid = alpha_get_thread_id();
		const ALPHA_INT local_m_s = partition[tid];
		const ALPHA_INT local_m_e = partition[tid + 1];
		tmp[tid] = (ALPHA_Number *)malloc(sizeof(ALPHA_Number) * b_rows * bs);
		memset(tmp[tid], 0, sizeof(ALPHA_Number) * b_rows * bs);
		
		if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
		{
			for (ALPHA_INT br = local_m_s; br < local_m_e; br++)
			{
				ALPHA_INT row = br * bs;
				ALPHA_INT block_start = A->rows_start[br],block_end = A->rows_end[br];
				ALPHA_INT lower_end = alpha_upper_bound(&A->col_indx[block_start],&A->col_indx[block_end],br)-A->col_indx;
				for(ALPHA_INT ai = block_start; ai < lower_end;ai++)
				{
					ALPHA_INT bc = A->col_indx[ai];
					ALPHA_INT col = bc * bs;
					ALPHA_INT a0_idx = ai * bs2;
					if (bc == br)
					{
						for (ALPHA_INT b_row = 0; b_row < bs; b_row++)
						{
							for (ALPHA_INT b_col = 0; b_col < b_row; b_col++)
							{
								alpha_madde(tmp[tid][b_row + row], A->values[a0_idx + b_row * bs + b_col], x[col + b_col]);
								alpha_madde(tmp[tid][b_col + col], A->values[a0_idx + b_row * bs + b_col], x[row + b_row]);
							}
						}
					}
					else
					{
						for (ALPHA_INT b_row = 0; b_row < bs; b_row++)
						{
							for (ALPHA_INT b_col = 0; b_col < bs; b_col++)
							{
								alpha_madde(tmp[tid][b_row + row], A->values[a0_idx + b_row * bs + b_col], x[col + b_col]);
								alpha_madde(tmp[tid][b_col + col], A->values[a0_idx + b_row * bs + b_col], x[row + b_row]);
							}
						}
					}
				}
			}
		}
		else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
		{
			for (ALPHA_INT br = 0; br < b_rows; br++)
			{
				ALPHA_INT row = br * bs;
				ALPHA_INT block_start = A->rows_start[br],block_end = A->rows_end[br];
				ALPHA_INT lower_end = alpha_upper_bound(&A->col_indx[block_start],&A->col_indx[block_end],br)-A->col_indx;
				for(ALPHA_INT ai = block_start; ai < lower_end;ai++)
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
							for (ALPHA_INT b_row = b_col + 1; b_row < bs; b_row++)
							{
								alpha_madde(tmp[tid][b_row + row],  A->values[a0_idx + b_col * bs + b_row], x[col + b_col]);
								alpha_madde(tmp[tid][b_col + col],  A->values[a0_idx + b_col * bs + b_row], x[row + b_row]);
							}
						}
					}
					else
					{
						for (ALPHA_INT b_col = 0; b_col < bs; b_col++)
						{
							for (ALPHA_INT b_row = 0; b_row < bs; b_row++)
							{
								alpha_madde(tmp[tid][b_row + row], A->values[a0_idx + b_col * bs + b_row], x[col + b_col]);
								alpha_madde(tmp[tid][b_col + col], A->values[a0_idx + b_col * bs + b_row], x[row + b_row]);
							}
						}
					}
				}
			}
		}
	}

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for (ALPHA_INT i = 0; i < b_cols * bs; ++i)
	{
		ALPHA_Number tmp_y;
		alpha_setzero(tmp_y);
		for (ALPHA_INT j = 0; j < thread_num; ++j)
		{
			alpha_add(tmp_y, tmp_y, tmp[j][i]);
		}
		alpha_mul(y[i], y[i], beta);
		alpha_madde(y[i], x[i], alpha);
		alpha_madde(y[i], tmp_y, alpha);
	}
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
	for (ALPHA_INT i = 0; i < thread_num; ++i)
	{
		free(tmp[i]);
	}
	free(tmp);

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
