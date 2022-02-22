#include "alphasparse/opt.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    const ALPHA_INT num_thread = alpha_get_thread_num(); 
    const ALPHA_INT bs = A->block_size;
    ALPHA_Number* diag=(ALPHA_Number*) alpha_malloc(A->rows*bs*sizeof(ALPHA_Number));
    const ALPHA_INT m = A->rows*bs;
    const ALPHA_INT n = A->cols*bs;
    memset(diag, '\0', m * sizeof(ALPHA_Number));

    const ALPHA_INT bs2 = bs * bs;
    const ALPHA_INT b_rows = m / bs;
    const ALPHA_INT b_cols = n / bs;
    const alphasparse_layout_t block_layout = A->block_layout;
    if(block_layout != ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        printf("layout not consistent!!!\n");
        exit(-1);
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(ALPHA_INT br = 0 ; br < b_rows; br++){
        for(ALPHA_INT ai = A->rows_start[br]; ai < A->rows_end[br]; ai++){

            ALPHA_INT bc = A->col_indx[ai];
            if(bc == br){
                for(ALPHA_INT b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(br,b_row,bs)] = A->values[ai * bs2 +  b_row *(bs + 1)];
                }
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        ALPHA_Number* temp = (ALPHA_Number*) alpha_malloc(bs*sizeof(ALPHA_Number));
        const ALPHA_INT y0_offset = out_y_col * ldy;
        const ALPHA_INT x0_offset = out_y_col * ldx;

        for (ALPHA_INT br = 0; br < b_rows; br++)
        {
            for(ALPHA_INT i = 0 ; i < bs ; i++){
                alpha_setzero(temp[i]);
            }
            ALPHA_INT diagBlock = -1;
            for (ALPHA_INT ai = A->rows_start[br]; ai < A->rows_end[br]; ai++)
            {
                ALPHA_INT bc = A->col_indx[ai];
                if(bc < br)
                    //col-major
                    for(ALPHA_INT col = 0; col < bs; col++)
                    {
                    //all entities belongs to upper triangle 
                        ALPHA_INT y_offset =  y0_offset + bc * bs + col;
                        ALPHA_INT a0_offset = ai * bs2 +  col * bs;
                        for(ALPHA_INT row = 0 ; row < bs ; row++)
                        {
                            ALPHA_INT ele_offset =  a0_offset + row;
                            alpha_madde(temp[row], A->values[ ele_offset ] ,y[y_offset]);
                        }
                    }
                //diagonal must be none-zero block
                if( bc==br ){
                    diagBlock = ai;
                }
            }
            if(diagBlock == -1)
            {
                printf("lhs matrix invalid for trsm!!!\n");
                exit(-1);
            }
            //col-major
            //top-left most
            for(ALPHA_INT col = 0; col < bs; col++)
            {
                //upper triangle of block
                ALPHA_Number t;
                alpha_setzero(t);
                alpha_mul(t,alpha,x[x0_offset + br * bs + col]);
                alpha_sub(t,t,temp[col]);
                alpha_div(y[y0_offset + br * bs + col],t,diag[col + br * bs]);

                for(ALPHA_INT row = col + 1; row < bs; row++){
                    alpha_madde(temp[row], A->values[ diagBlock * bs2 +  col * bs + row],y[y0_offset + br * bs + col ]);
                }
            }
        }
        alpha_free(temp);
    }
    alpha_free(diag);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
