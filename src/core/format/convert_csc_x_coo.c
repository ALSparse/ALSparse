#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

static int col_first_cmp(const ALPHA_Point *a, const ALPHA_Point *b)
{
    if (a->y != b->y)
        return a->y - b->y;
    return a->x - b->x;
}

alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_CSC **dest)
{
    ALPHA_SPMAT_CSC *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSC));
    *dest = mat;
    ALPHA_INT m = source->rows;
    ALPHA_INT n = source->cols;
    ALPHA_INT nnz = source->nnz;
    //sort by (col,row)
    ALPHA_Point *points = alpha_malloc(sizeof(ALPHA_Point) * nnz);
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        points[i].x = source->row_indx[i];
        points[i].y = source->col_indx[i];
        points[i].v = source->values[i];
    }
    qsort(points, nnz, sizeof(ALPHA_Point), (__compar_fn_t)col_first_cmp);
    mat->rows = m;
    mat->cols = n;
    ALPHA_INT *cols_offset = alpha_memalign((n + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->row_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    mat->cols_start = cols_offset;
    mat->cols_end = cols_offset + 1;
    mat->cols_start[0] = 0;
    ALPHA_INT index = 0;
    ALPHA_INT count = 0;
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        while (index < points[i].y)
        {
            mat->cols_end[index] = count;
            index += 1;
        }
        if (index == points[i].y)
        {
            count += 1;
        }
    }
    while (index < n - 1)
    {
        mat->cols_end[index] = count;
        index += 1;
    }
    mat->cols_end[n - 1] = count;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT* partition = alpha_malloc(sizeof(ALPHA_INT)*(num_threads+1));
    balanced_partition_row_by_nnz(mat->cols_end, mat->cols, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT lcs = partition[tid];
        ALPHA_INT lch = partition[tid + 1];
        for (ALPHA_INT ac = lcs; ac < lch; ac++)
        {
            for (ALPHA_INT ai = mat->cols_start[ac]; ai < mat->cols_end[ac]; ++ai)
            {
                mat->row_indx[ai] = points[ai].x;
                mat->values[ai] = points[ai].v;
            }
        }
    }
    alpha_free(points);
    alpha_free(partition);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}