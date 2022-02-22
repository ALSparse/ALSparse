#include "alphasparse/kernel.h"
#include "stdio.h"
#include "alphasparse/util.h"
alphasparse_status_t
ONAME(const ALPHA_Number alpha,
		              const ALPHA_SPMAT_BSR *A,
		              const ALPHA_Number *x,
		              const ALPHA_Number beta,
		              ALPHA_Number *y)
{
	ALPHA_INT bs = A->block_size;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	
	ALPHA_Number temp;
	alpha_setzero(temp);
	for (ALPHA_INT j = 0; j < A->rows * A->block_size; j++){
		alpha_mul(y[j], y[j], beta);
	}
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
    	for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_indx[ai];
    	        if(col > i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first diag indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 <= s + s/bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_mul(temp, alpha, cv);
							alpha_mul(temp, temp, x[s1-s+col*bs]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], temp);
							if(s1 != s+s/bs) {
								alpha_mul(temp, alpha, cv);
								alpha_mul(temp, temp, x[m_s+s/bs]);
								alpha_add(y[s1-s+col*bs], y[s1-s+col*bs], temp);
							}
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_mul(temp, alpha, cv);
							alpha_mul(temp, temp, x[s1-s+col*bs]);
							alpha_add(y[m_s+s/bs], y[m_s+s/bs], temp);
							alpha_mul(temp, alpha, cv);
							alpha_mul(temp, temp, x[m_s+s/bs]);
							alpha_add(y[s1-s+col*bs], y[s1-s+col*bs], temp);
						}
					}
    	        }
    	    }
    	}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
		for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->rows_start[i]; ai < A->rows_end[i]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_indx[ai];
    	        if(col > i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s +s/bs; s1 < s + bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_mul(temp, alpha, cv);
							alpha_mul(temp, temp, x[col*bs+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], temp);
							if(s1 != s+s/bs) {
								alpha_mul(temp, alpha, cv);
								alpha_mul(temp, temp, x[m_s+s1-s]);
								alpha_add(y[col*bs+s/bs], y[col*bs+s/bs], temp);
							}
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							ALPHA_Number cv = A->values[s1+ai*bs*bs];
							alpha_conj(cv, cv);
							alpha_mul(temp, alpha, cv);
							alpha_mul(temp, temp, x[col*bs+s/bs]);
							alpha_add(y[m_s+s1-s], y[m_s+s1-s], temp);
							alpha_mul(temp, alpha, cv);
							alpha_mul(temp, temp, x[m_s+s1-s]);
							alpha_add(y[col*bs+s/bs], y[col*bs+s/bs], temp);
						}
					}
    	        }
    	    }
    	}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
