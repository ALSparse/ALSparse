#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

static int row_first_cmp(const ALPHA_Point *a, const ALPHA_Point *b)
{
    if (a->x != b->x)
        return a->x - b->x;
    return a->y - b->y;
}

alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_CSR **dest)
{
    ALPHA_SPMAT_CSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSR));
    *dest = mat;
    ALPHA_INT m = source->rows;
    ALPHA_INT n = source->cols;
    ALPHA_INT nnz = source->nnz;
    // sort by (row,col)
    ALPHA_Point *points = alpha_malloc(sizeof(ALPHA_Point) * nnz);
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        points[i].x = source->row_indx[i];
        points[i].y = source->col_indx[i];
        points[i].v = source->values[i];
    }
    qsort(points, nnz, sizeof(ALPHA_Point), (__compar_fn_t)row_first_cmp);
    mat->rows = m;
    mat->cols = n;
    ALPHA_INT *rows_offset = alpha_memalign((m + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->col_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    mat->rows_start = rows_offset;
    mat->rows_end = rows_offset + 1;
    mat->rows_start[0] = 0;
    ALPHA_INT index = 0;
    ALPHA_INT count = 0;
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        while (index < points[i].x)
        {
            mat->rows_end[index] = count;
            index += 1;
        }
        if (index == points[i].x)
        {
            count += 1;
        }
    }
    while (index < m - 1)
    {
        mat->rows_end[index] = count;
        index += 1;
    }
    mat->rows_end[m - 1] = count;

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(mat->rows_end, mat->rows, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT lrs = partition[tid];
        ALPHA_INT lrh = partition[tid + 1];
        for (ALPHA_INT ar = lrs; ar < lrh; ar++)
        {
            for (ALPHA_INT ai = mat->rows_start[ar]; ai < mat->rows_end[ar]; ++ai)
            {
                mat->col_indx[ai] = points[ai].y;
                mat->values[ai] = points[ai].v;
            }
        }
    }
    alpha_free(points);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
