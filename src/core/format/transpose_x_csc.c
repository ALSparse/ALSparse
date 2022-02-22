#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *A, ALPHA_SPMAT_CSC **B)
{
    ALPHA_SPMAT_CSC *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSC));
    *B = mat;
    ALPHA_INT rowA = A->rows;
    ALPHA_INT colA = A->cols;
    mat->rows = colA;
    mat->cols = rowA;
    ALPHA_INT nnz = A->cols_end[colA - 1];
    ALPHA_INT *cols_offset = alpha_memalign((mat->cols + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->cols_start = cols_offset;
    mat->cols_end = cols_offset + 1;
    mat->row_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    ALPHA_INT row_counter[rowA];
    ALPHA_INT col_offset[mat->cols];
    memset(row_counter, '\0', rowA * sizeof(ALPHA_INT));
    for (ALPHA_INT i = 0; i < nnz; ++i)
    {
        row_counter[A->row_indx[i]] += 1;
    }
    col_offset[0] = 0;
    mat->cols_start[0] = 0;
    for (ALPHA_INT i = 1; i < mat->cols; ++i)
    {
        col_offset[i] = col_offset[i - 1] + row_counter[i - 1];
        mat->cols_end[i - 1] = col_offset[i];
    }
    mat->cols_end[mat->cols - 1] = nnz;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(mat->cols_end, mat->cols, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT lcs = partition[tid];
        ALPHA_INT lch = partition[tid + 1];
        for (ALPHA_INT ac = 0; ac < colA; ++ac)
        {
            for (ALPHA_INT ai = A->cols_start[ac]; ai < A->cols_end[ac]; ++ai)
            {
                ALPHA_INT bc = A->row_indx[ai];
                if (bc < lcs || bc >= lch)
                    continue;
                ALPHA_INT index = col_offset[bc];
                mat->row_indx[index] = ac;
                mat->values[index] = A->values[ai];
                col_offset[bc] += 1;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}