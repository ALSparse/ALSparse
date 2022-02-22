#include "alphasparse/kernel_plain.h"
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
	m_inner = ( m_inner*bs == A->rows)?(m_inner):(m_inner+1);
	ALPHA_INT n_inner = A->cols/bs;
	n_inner = ( n_inner*bs == A->cols)?(n_inner):(n_inner+1);
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	ALPHA_Number *part = (ALPHA_Number*)malloc(A->rows*sizeof(ALPHA_Number));
	memset(part, 0, A->rows*sizeof(ALPHA_Number));

	 ALPHA_Number tmp;
	alpha_setzero(tmp);
	for (ALPHA_INT j = 0; j < A->rows; j++){
		alpha_mul(part[j], y[j], beta);
		//part[j] = y[j]*beta;
		alpha_setzero(y[j]);
	}
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		ALPHA_INT diag_block = 0;
    	for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_indx[ai];
    	        if(col < i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first diag indx of the s-row in bolck[ai][col]
						// of A->value
						alpha_add(y[m_s+s/bs], y[m_s+s/bs], x[s/bs+col*bs]);
						//y[m_s+s/bs] += x[s/bs+col*bs];
						for (ALPHA_INT s1 = s + s/bs + 1 ; s1 < s + bs; s1++){
							alpha_madde(y[m_s+s/bs], A->values[s1+ai*bs*bs], x[s1-s+col*bs]);
							//y[m_s+s/bs] += A->values[s1+ai*bs*bs]*x[s1-s+col*bs];
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][col*bs+s1-ai*bs*bs-s]
							alpha_madde(y[m_s+s/bs], A->values[s1+ai*bs*bs], x[s1-s+col*bs]);
							//y[m_s+s/bs] += A->values[s1+ai*bs*bs]*x[s1-s+col*bs];
						}
					}
    	        }
    	    }if (diag_block == 0){
				for (ALPHA_INT st = 0; st < bs; st++){
					alpha_add(y[m_s+st], y[m_s+st], x[st+m_s]);
				}
			}
    	}
		for(ALPHA_INT k = 0; k < A->rows; k++){
			alpha_madde(part[k], alpha, y[k]);
			//y[k] = part[k];
			y[k] = part[k];
			//y[k] = y[k]*alpha + part[k];
		}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
		ALPHA_INT diag_block = 0;
		for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_indx[ai];
    	        if(col < i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						alpha_add(y[m_s+s/bs], alpha, x[s/bs+col*bs]);
						//y[m_s+s/bs] += alpha*x[s/bs+col*bs];
						for (ALPHA_INT s1 = s ; s1 < s + s/bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][(i+ai)*bs+s/bs]
							alpha_mul(tmp, alpha, A->values[s1+ai*bs*bs]);
							alpha_mul(tmp, tmp, x[col*bs+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], tmp);
							//y[m_s+s1-s] += alpha*A->values[s1+ai*bs*bs]*x[col*bs+s/bs];
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							alpha_mul(tmp, alpha, A->values[s1+ai*bs*bs]);
							alpha_mul(tmp, tmp, x[col*bs+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], tmp);
							//y[m_s+s1-s] += alpha*A->values[s1+ai*bs*bs]*x[col*bs+s/bs];
						}
					}
    	        }
    	    }
            if (diag_block == 0){
				for (ALPHA_INT st = 0; st < bs; st++){
					alpha_madde(y[m_s+st], alpha, x[st+m_s]);
					//y[m_s+st] += alpha*x[st+m_s];
				}
			}
    	}
        for(ALPHA_INT k = 0; k < A->rows; k++){
			alpha_madde(part[k], y[k], alpha);
			y[k] = part[k];
		    //y[k] = y[k]*alpha + part[k];
	    }
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
