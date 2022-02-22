#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_BSR **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout)
{
    ALPHA_INT m = source->rows;
    ALPHA_INT n = source->cols;
    ALPHA_INT nnz = source->nnz;
    if (m % block_size != 0 || n % block_size != 0)
    {
        printf("in convert_bsr_coo , rows or cols is not divisible by block_size!!!");
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    ALPHA_SPMAT_BSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_BSR));
    *dest = mat;
    ALPHA_INT block_rows = m / block_size;
    ALPHA_INT block_cols = n / block_size;
    ALPHA_INT *block_row_offset = alpha_memalign((block_rows + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->rows = block_rows;
    mat->cols = block_cols;
    mat->block_size = block_size;
    mat->block_layout = block_layout;
    mat->rows_start = block_row_offset;
    mat->rows_end = block_row_offset + 1;
    ALPHA_SPMAT_CSR *csr;
    check_error_return(convert_csr_coo(source, &csr));
    ALPHA_INT *pos;
    ALPHA_INT bcl, ldp;
    csr_col_partition(csr, 0, m, block_size, &pos, &bcl, &ldp);
    mat->rows_start[0] = 0;
    ALPHA_INT block_nnz = 0;
    for (ALPHA_INT br = 0, brs = 0; brs < m; br += 1, brs += block_size)
    {
        ALPHA_INT bre = brs + block_size;
        for (ALPHA_INT bi = 0; bi < bcl; bi++)
        {
            bool has_non_zero = false;
            for (ALPHA_INT r = brs; r < bre; r++)
            {
                if (pos[index2(r, bi + 1, ldp)] - pos[index2(r, bi, ldp)] > 0)
                {
                    has_non_zero = true;
                    break;
                }
            }
            if (has_non_zero)
            {
                block_nnz += 1;
            }
        }
        mat->rows_end[br] = block_nnz;
    }
    mat->col_indx = alpha_memalign(block_nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(block_nnz * block_size * block_size * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT* partition = alpha_malloc(sizeof(ALPHA_INT)*(num_threads+1));
    balanced_partition_row_by_nnz(mat->rows_end, block_rows, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT lrs = partition[tid];
        ALPHA_INT lrh = partition[tid + 1];
        ALPHA_INT *col_indx = &mat->col_indx[mat->rows_start[lrs]];
        ALPHA_Number *values = &mat->values[mat->rows_start[lrs] * block_size * block_size];
        ALPHA_INT count = mat->rows_end[lrh - 1] - mat->rows_start[lrs];
        ALPHA_INT index = 0;
        memset(values, '\0', count * block_size * block_size * sizeof(ALPHA_Number));
        for (ALPHA_INT brs = lrs * block_size; brs < lrh * block_size; brs += block_size)
        {
            ALPHA_INT bre = alpha_min(brs + block_size, lrh * block_size);
            for (ALPHA_INT bi = 0; bi < bcl; bi++)
            {
                bool has_non_zero = false;
                for (ALPHA_INT r = brs; r < bre; r++)
                {
                    if (pos[index2(r, bi + 1, ldp)] - pos[index2(r, bi, ldp)] > 0)
                    {
                        has_non_zero = true;
                        break;
                    }
                }
                if (has_non_zero)
                {
                    col_indx[index] = bi;
                    ALPHA_Number *block_values = values + index * block_size * block_size;
                    if (block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        for (ALPHA_INT r = brs; r < bre; r++)
                        {
                            for (ALPHA_INT ai = pos[index2(r, bi, ldp)]; ai < pos[index2(r, bi + 1, ldp)]; ai++)
                            {
                                ALPHA_INT ac = csr->col_indx[ai];
                                block_values[ac - bi * block_size] = csr->values[ai];
                            }
                            block_values += block_size;
                        }
                    }
                    else
                    {
                        for (ALPHA_INT r = brs; r < bre; r++)
                        {
                            ALPHA_INT block_row_index = r - brs;
                            for (ALPHA_INT ai = pos[index2(r, bi, ldp)]; ai < pos[index2(r, bi + 1, ldp)]; ai++)
                            {
                                ALPHA_INT ac = csr->col_indx[ai];
                                ALPHA_INT block_col_index = ac - bi * block_size;
                                block_values[index2(block_col_index, block_row_index, block_size)] = csr->values[ai];
                            }
                        }
                    }
                    index += 1;
                }
            }
        }
    }
    destroy_csr(csr);
    alpha_free(pos);
    alpha_free(partition);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
