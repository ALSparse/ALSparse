#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *A, ALPHA_SPMAT_BSR **B)
{
    ALPHA_SPMAT_BSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_BSR));
    *B = mat;
    ALPHA_INT block_size = A->block_size;
    ALPHA_INT rowA = A->rows * block_size;
    ALPHA_INT colA = A->cols * block_size;
    ALPHA_INT block_rowA = A->rows;
    ALPHA_INT block_colA = A->cols;
    mat->rows = A->cols;
    mat->cols = A->rows;
    mat->block_size = block_size;
    mat->block_layout = A->block_layout;
    ALPHA_INT block_nnz = A->rows_end[block_rowA-1];
    ALPHA_INT *rows_offset = alpha_memalign((block_colA + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->rows_start = rows_offset;
    mat->rows_end = rows_offset + 1;
    mat->col_indx = alpha_memalign(block_nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(block_nnz * block_size * block_size * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    ALPHA_INT col_counter[block_colA];
    ALPHA_INT row_offset[block_colA];
    memset(col_counter, '\0', block_colA * sizeof(ALPHA_INT));
    for (ALPHA_INT i = 0; i < block_nnz; ++i)
    {
        col_counter[A->col_indx[i]] += 1;
    }
    row_offset[0] = 0;
    mat->rows_start[0] = 0;
    for (ALPHA_INT i = 1; i < block_colA; ++i)
    {
        row_offset[i] = row_offset[i - 1] + col_counter[i - 1];
        mat->rows_end[i - 1] = row_offset[i];
    }
    mat->rows_end[block_colA - 1] = block_nnz;
    for (ALPHA_INT r = 0; r < block_rowA; ++r)
    {
        for (ALPHA_INT ai = A->rows_start[r]; ai < A->rows_end[r]; ++ai)
        {
            ALPHA_INT ac = A->col_indx[ai];
            ALPHA_INT index = row_offset[ac];
            mat->col_indx[index] = r;
            const ALPHA_Number* A_values = A->values + ai * block_size * block_size;
            ALPHA_Number* B_values = mat->values + index * block_size * block_size;
            for(ALPHA_INT br = 0;br < block_size;++br){
                for(ALPHA_INT bc = 0;bc < block_size;++bc){
                    B_values[index2(bc,br,block_size)] = A_values[index2(br,bc,block_size)];       
                }
            }
            row_offset[ac] += 1;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
