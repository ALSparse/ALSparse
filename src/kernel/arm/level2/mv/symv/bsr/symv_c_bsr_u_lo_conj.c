#include "alphasparse/kernel.h"
#include "stdio.h"
#include <stdlib.h>
#include <string.h>
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_BSR *A,
		              const ALPHA_Number *x,
		              const ALPHA_Number beta,
		              ALPHA_Number *y)
{
	ALPHA_INT bs = A->block_size;
	ALPHA_INT m_inner = A->rows/bs;
	ALPHA_INT n_inner = A->cols/bs;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	ALPHA_Number *part = (ALPHA_Number*)malloc(A->rows*sizeof(ALPHA_Number));
	memset(part, 0, A->rows*sizeof(ALPHA_Number));

	for (ALPHA_INT j = 0; j < A->rows; j++){
		alpha_mul(part[j], y[j], beta);
		y[j].imag = 0.0f;
		y[j].real = 0.0f;
	}
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		ALPHA_INT diag_block = 0;
    	for(int i = 0; i < m_inner; ++i)
    	{
			int m_s = i*bs;
    	    for(int ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
    	    {
    	        const int col = A->col_indx[ai];
    	        if(col < i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					diag_block = 1;
					for (int s = 0; s < bs*bs; s=s+bs){
						alpha_add(y[m_s+s/bs], y[m_s+s/bs], x[s/bs+col*bs]);
						for (int s1 = s + s/bs + 1 ; s1 < s + bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_madde(y[m_s+s/bs], cv, x[s1-s+col*bs]);
							alpha_madde(y[s1-s+col*bs], cv, x[m_s+s/bs]);
						}
					}
    	        }
    	        else
    	        {
					for (int s = 0; s < bs*bs; s=s+bs){
						for (int s1 = s; s1 < s+bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_madde(y[m_s+s/bs], cv, x[s1-s+col*bs]);
							alpha_madde(y[s1-s+col*bs], cv, x[m_s+s/bs]);
						}
					}
    	        }
    	    }if (diag_block == 0){
				for (int st = 0; st < bs; st++){
					alpha_add(y[m_s+st], y[m_s+st], x[st+m_s]);
				}
			}
    	}

		for(ALPHA_INT k = 0; k < A->rows; k++){
			alpha_madde(part[k], y[k], alpha);
			y[k] = part[k];
		}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
		ALPHA_INT diag_block = 0;
		for(int i = 0; i < m_inner; ++i)
    	{
			int m_s = i*bs;
    	    for(int ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
    	    {
    	        const int col = A->col_indx[ai];
    	        if(col < i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					diag_block = 1;
					for (int s = 0; s < bs*bs; s=s+bs){
						alpha_madde(y[m_s+s/bs], alpha, x[s/bs+col*bs]);
						for (int s1 = s ; s1 < s + s/bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_mul(y[m_s+s1-s], alpha, cv);
							alpha_mul(y[m_s+s1-s], y[m_s+s1-s], x[col*bs+s/bs]);
							alpha_mul(y[col*bs+s/bs], alpha, cv);
							alpha_mul(y[col*bs+s/bs], y[col*bs+s/bs], x[m_s+s1-s]);
						}
					}
    	        }
    	        else
    	        {
					for (int s = 0; s < bs*bs; s=s+bs){
						for (int s1 = s; s1 < s+bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_mul(y[m_s+s1-s], alpha, cv);
							alpha_mul(y[m_s+s1-s], y[m_s+s1-s], x[col*bs+s/bs]);
							alpha_mul(y[col*bs+s/bs], alpha, cv);
							alpha_mul(y[col*bs+s/bs], y[col*bs+s/bs], x[m_s+s1-s]);
						}
					}
    	        }
    	    }if (diag_block == 0){
				for (int st = 0; st < bs; st++){
					alpha_madde(y[m_s+st], alpha, x[st+m_s]);
				}
			}
    	}for(ALPHA_INT k = 0; k < A->rows; k++){
			alpha_add(y[k], y[k], part[k]);
		}
	}
	else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
