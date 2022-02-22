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
#ifdef COMPLEX
	ALPHA_INT bs = A->block_size;
	ALPHA_INT m_inner = A->rows/bs;
	ALPHA_INT n_inner = A->cols/bs;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	ALPHA_Number *part = (ALPHA_Number*)malloc(A->rows*sizeof(ALPHA_Number));
	memset(part, 0, A->rows*sizeof(ALPHA_Number));

	for (ALPHA_INT j = 0; j < A->rows; j++){
		alpha_mul(part[j], y[j], beta);
		//prALPHA_INTf("part[%d].real=%f\tpart[%d].imag=%f\n",j,part[j].real,j,part[j].imag);
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
						alpha_add(y[m_s+s/bs], y[m_s+s/bs], x[s/bs+col*bs]);
						//y[m_s+s/bs] += x[s/bs+col*bs];
						for (ALPHA_INT s1 = s + s/bs + 1 ; s1 < s + bs; s1++){
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_madde(y[m_s+s/bs], val, x[s1-s+col*bs]);
							//y[m_s+s/bs] += A->values[s1+ai*bs*bs]*x[s1-s+col*bs];
							alpha_madde(y[s1-s+col*bs], val, x[m_s+s/bs]);
							//y[s1-s+col*bs] += A->values[s1+ai*bs*bs]*x[m_s+s/bs];
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_madde(y[m_s+s/bs], val, x[s1-s+col*bs]);
							//y[m_s+s/bs] += A->values[s1+ai*bs*bs]*x[s1-s+col*bs];
							alpha_madde(y[s1-s+col*bs], val, x[m_s+s/bs]);
							//y[s1-s+col*bs] += A->values[s1+ai*bs*bs]*x[m_s+s/bs];
						}
					}
    	        }
    	    }if (diag_block == 0){
				for (ALPHA_INT st = 0; st < bs; st++){
					alpha_add(y[m_s+st], y[m_s+st], x[st+m_s]);
					//y[m_s+st] += x[st+m_s];
				}
			}
    	}
		//for (ALPHA_INT k = 0; k < A->rows; k++){
			//prALPHA_INTf("part[%d].real=%f\tpart[%d].imag=%f\n",k,part[k].real,k,part[k].imag);
			//prALPHA_INTf("y[%d].real=%f\ty[%d].imag=%f\n",k,y[k].real,k,y[k].imag);
		//}
		for(ALPHA_INT k = 0; k < A->rows; k++){
			//prALPHA_INTf("=======================================\n");
			//prALPHA_INTf("part[%d].real=%f\tpart[%d].imag=%f\n",k,part[k].real,k,part[k].imag);
			//prALPHA_INTf("y[%d].real=%f\ty[%d].imag=%f\n",k,y[k].real,k,y[k].imag);
			alpha_madde(part[k], y[k], alpha);
			//y[k] = part[k];
			y[k] = part[k];
			//prALPHA_INTf("part[%d].real=%f\tpart[%d].imag=%f\n",k,part[k].real,k,part[k].imag);
			//prALPHA_INTf("y[%d].real=%f\ty[%d].imag=%f\n",k,y[k].real,k,y[k].imag);
			//prALPHA_INTf("=======================================\n");
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
						alpha_madde(y[m_s+s/bs], alpha, x[s/bs+col*bs]);
						//y[m_s+s/bs] += alpha*x[s/bs+col*bs];
						for (ALPHA_INT s1 = s ; s1 < s + s/bs; s1++){
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_mul(y[m_s+s1-s], alpha, val);
							alpha_mul(y[m_s+s1-s], y[m_s+s1-s], x[col*bs+s/bs]);
							//y[m_s+s1-s] += alpha*A->values[s1+ai*bs*bs]*x[col*bs+s/bs];
							alpha_mul(y[col*bs+s/bs], alpha,val);
							alpha_mul(y[col*bs+s/bs], y[col*bs+s/bs], x[m_s+s1-s]);
							//y[col*bs+s/bs] += alpha*A->values[s1+ai*bs*bs]*x[m_s+s1-s];
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							ALPHA_Number val =  A->values[s1+ai*bs*bs];
							alpha_conj(val,val);
							alpha_mul(y[m_s+s1-s], alpha, val);
							alpha_mul(y[m_s+s1-s], y[m_s+s1-s], x[col*bs+s/bs]);
							//y[m_s+s1-s] += alpha*A->values[s1+ai*bs*bs]*x[col*bs+s/bs];
							alpha_mul(y[col*bs+s/bs], alpha, val);
							alpha_mul(y[col*bs+s/bs], y[col*bs+s/bs], x[m_s+s1-s]);
							//y[col*bs+s/bs] += alpha*A->values[s1+ai*bs*bs]*x[m_s+s1-s];
						}
					}
    	        }
    	    }if (diag_block == 0){
				for (ALPHA_INT st = 0; st < bs; st++){
					alpha_madde(y[m_s+st], alpha, x[st+m_s]);
					//y[m_s+st] += alpha*x[st+m_s];
				}
			}
    	}for(ALPHA_INT k = 0; k < A->rows; k++){
			alpha_add(y[k], y[k], part[k]);
			//y[k] = y[k] + part[k];
		}
	}
	else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
#else
	return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
}
