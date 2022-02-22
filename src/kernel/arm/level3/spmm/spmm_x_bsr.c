#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <stdbool.h>
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *A, const ALPHA_SPMAT_BSR *B, ALPHA_SPMAT_BSR **matC)
{
    check_return(A->cols != B->rows, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(A->block_size != B->block_size, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(A->block_layout != B->block_layout, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    ALPHA_SPMAT_BSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_BSR));
    *matC = mat;
    mat->rows         = A->rows;
    mat->cols         = B->cols;
    mat->block_layout = A->block_layout;
    mat->block_size   = A->block_size;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    ALPHA_INT bs = A->block_size;
    bool *flag = alpha_memalign(sizeof(bool) * n, DEFAULT_ALIGNMENT);
    ALPHA_INT nnz = 0;
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        memset(flag, '\0', sizeof(bool) * n);
        for (ALPHA_INT ai = A->rows_start[ar]; ai < A->rows_end[ar]; ai++)
        {
            ALPHA_INT br = A->col_indx[ai];
            for (ALPHA_INT bi = B->rows_start[br]; bi < B->rows_end[br]; bi++)
            {
                if (!flag[B->col_indx[bi]])
                {
                    nnz += 1;
                    flag[B->col_indx[bi]] = true;
                }		        
            }
        }
    }
    alpha_free(flag);

    ALPHA_INT *row_offset = alpha_memalign(sizeof(ALPHA_INT) * (m + 1), DEFAULT_ALIGNMENT);
    mat->rows_start = row_offset;
    mat->rows_end = row_offset + 1;
    mat->col_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * bs * bs * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);

    ALPHA_Number *values = alpha_memalign(sizeof(ALPHA_Number) * n * bs * bs, DEFAULT_ALIGNMENT);

    ALPHA_INT index = 0;
    mat->rows_start[0] = 0;
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        bool *flaggg = alpha_memalign(sizeof(bool) * n, DEFAULT_ALIGNMENT); 
	    memset(flaggg, '\0', sizeof(bool) * n);
	    memset(values, '\0', sizeof(ALPHA_Number) * n * bs * bs);
        for (ALPHA_INT ai = A->rows_start[ar]; ai < A->rows_end[ar]; ai++)
        {
            ALPHA_INT br = A->col_indx[ai];
            for (ALPHA_INT bi = B->rows_start[br]; bi < B->rows_end[br]; bi++)
            {
		        ALPHA_INT bc = B->col_indx[bi];
                if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                {
                    // row major
                    for(ALPHA_INT block_ar = 0; block_ar < A->block_size; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < A->block_size; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < B->block_size; block_bc++)
                            {
                                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                ALPHA_Number av = A->values[bs*bs*ai + bs*block_ar + block_ac];
                                ALPHA_Number bv = B->values[bs*bs*bi + bs*block_br + block_bc];
                                alpha_madde(values[bc*bs*bs + block_ar*bs + block_bc], av, bv);
				                flaggg[B->col_indx[bi]]=true;
                            }
                        }
                    }
                }
                else
                {
                    // col major
		            for(ALPHA_INT block_ar = 0; block_ar < A->block_size; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < A->block_size; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < B->block_size; block_bc++)
                            {
                                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                ALPHA_Number av = A->values[bs*bs*ai + bs*block_ac + block_ar];
                                ALPHA_Number bv = B->values[bs*bs*bi + bs*block_bc + block_br];
                                alpha_madde(values[bc*bs*bs + block_bc*bs + block_ar], av, bv);
                                flaggg[B->col_indx[bi]]=true;
			                }
			            }
		            }
                }
            }
        }
        for (ALPHA_INT c = 0; c < n; c++)
        {
	        if(flaggg[c] == true)
            {
                mat->col_indx[index] = c;
                for(int i=0; i < bs*bs; i++)
                {
                    mat->values[index*bs*bs+i] = values[c*bs*bs+i];
                }
                index += 1;
            }
        }
        mat->rows_end[ar] = index;
    }

    alpha_free(values);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
