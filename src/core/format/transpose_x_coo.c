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

alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *s, ALPHA_SPMAT_COO **d)
{
    ALPHA_INT nnz = s->nnz;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_Point *points = alpha_memalign(nnz * sizeof(ALPHA_Point), DEFAULT_ALIGNMENT);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 0; i < nnz; ++i)
    {
        points[i].x = s->col_indx[i];
        points[i].y = s->row_indx[i];
        points[i].v = s->values[i];
    }
    qsort(points, nnz, sizeof(ALPHA_Point), (__compar_fn_t)row_first_cmp);
    ALPHA_SPMAT_COO *mat = alpha_malloc(sizeof(ALPHA_SPMAT_COO));
    *d = mat;
    mat->rows = s->cols;
    mat->cols = s->rows;
    mat->nnz = s->nnz;
    mat->row_indx = alpha_memalign(sizeof(ALPHA_INT) * nnz, DEFAULT_ALIGNMENT);
    mat->col_indx = alpha_memalign(sizeof(ALPHA_INT) * nnz, DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(sizeof(ALPHA_Number) * nnz, DEFAULT_ALIGNMENT);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        mat->row_indx[i] = points[i].x;
        mat->col_indx[i] = points[i].y;
        mat->values[i] = points[i].v;
    }
    alpha_free(points);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}