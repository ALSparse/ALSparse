#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *matA, const ALPHA_SPMAT_BSR *matB, ALPHA_Number *matC, const ALPHA_INT ldc)
{
    if (matA->cols != matB->rows || ldc < matB->cols)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    if(matA->block_layout != matB->block_layout)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    if(matA->block_size != matB->block_size) 
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
 
    ALPHA_INT bs = matA->block_size;
    ALPHA_INT m = matA->rows * bs;
    ALPHA_INT n = matB->cols * bs;
    
    // init C
    for(ALPHA_INT i = 0; i < m; i++)
        for(ALPHA_INT j = 0; j < n; j++)
        {
            alpha_setzero(matC[index2(i, j, ldc)]);
        }
    
    ALPHA_INT A_block_cols = matA->cols;
    ALPHA_INT A_block_rows = matA->rows;
    ALPHA_INT B_block_cols = matB->cols;
    ALPHA_INT B_block_rows = matB->rows;
    // 计算
    for (ALPHA_INT ar = 0; ar < A_block_rows; ar++)
    {
        for (ALPHA_INT ai = matA->rows_start[ar]; ai < matA->rows_end[ar]; ai++)
        {
            ALPHA_INT br = matA->col_indx[ai];
            //ALPHA_Number av = matA->values[ai];// av：matA->values[block_size*block_size*ai, block_size*block_size*ai+block_size*block_size]
            for (ALPHA_INT bi = matB->rows_start[br]; bi < matB->rows_end[br]; bi++)
            {
                ALPHA_INT bc = matB->col_indx[bi];
                //ALPHA_Number bv = matB->values[bi]; //bv：matB->values[block_size*block_size*bi: block_size*block_size*bi+block_size*block_size]
                // 块内再做一个稠密矩阵乘法
                //matC[index2(ar, bc, ldc)] += av * bv;
                if(matA->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                {
                    //printf("rowlayout");
		            // row major
                    for(ALPHA_INT block_ar = 0; block_ar < matA->block_size; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < matA->block_size; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < matB->block_size; block_bc++)
                            {
				                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                ALPHA_INT bs = matA->block_size;
                                ALPHA_Number av = matA->values[bs*bs*ai + bs*block_ar + block_ac];
                                ALPHA_Number bv = matB->values[bs*bs*bi + bs*block_br + block_bc];
                                //printf("%lf\t%lf\n", (ALPHA_Number)av, (ALPHA_Number)bv);
                                //matC[index2(ar*bs+block_ar, bc*bs+block_bc, ldc)] += av*bv;
                                alpha_madde(matC[index2(ar*bs+block_ar, bc*bs+block_bc, ldc)], av, bv);
			                }
                        }
                    }
                }
                else 
                {
                    //col major
                    for(ALPHA_INT block_ar = 0; block_ar < matA->block_size; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < matA->block_size; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < matB->block_size; block_bc++)
                            {
				                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                ALPHA_INT bs = matA->block_size;
                                ALPHA_Number av = matA->values[bs*bs*ai + bs*block_ac + block_ar];
                                ALPHA_Number bv = matB->values[bs*bs*bi + bs*block_bc + block_br];
                                //printf("%lf\t%lf\n", (ALPHA_Number)av, (ALPHA_Number)bv);
                                //matC[index2(ar*bs+block_ar, bc*bs+block_bc, ldc)] += av*bv;
                                alpha_madde(matC[index2(ar*bs+block_ar, bc*bs+block_bc, ldc)], av, bv);
			                }
                        }
                    }
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
