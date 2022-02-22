#include "alphasparse.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

alphasparse_status_t ONAME(alphasparse_matrix_t *A,
                          const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                          const ALPHA_INT rows,
                          const ALPHA_INT cols,
                          const ALPHA_INT nnz,
                          ALPHA_INT *row_indx,
                          ALPHA_INT *col_indx,
                          ALPHA_Number *values)
{
    alphasparse_matrix* AA = alpha_malloc(sizeof(alphasparse_matrix));
    *A = AA;
    ALPHA_SPMAT_COO *mat = alpha_malloc(sizeof(ALPHA_SPMAT_COO));
    AA->format = ALPHA_SPARSE_FORMAT_COO;
    AA->datatype = ALPHA_SPARSE_DATATYPE;
    AA->mat = mat;
    mat->rows = rows;
    mat->cols = cols;
    mat->nnz = nnz;
    mat->row_indx = alpha_memalign(sizeof(ALPHA_INT) * nnz, DEFAULT_ALIGNMENT);
    mat->col_indx = alpha_memalign(sizeof(ALPHA_INT) * nnz, DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(sizeof(ALPHA_Number) * nnz, DEFAULT_ALIGNMENT);
    if (indexing == ALPHA_SPARSE_INDEX_BASE_ZERO)
    {
        for (ALPHA_INT i = 0; i < nnz; ++i)
        {
            mat->row_indx[i] = row_indx[i];
            mat->col_indx[i] = col_indx[i];
            mat->values[i] = values[i];
        }
    }
    else
    {
        for (ALPHA_INT i = 0; i < nnz; ++i)
        {
            mat->row_indx[i] = row_indx[i] - 1;
            mat->col_indx[i] = col_indx[i] - 1;
            mat->values[i] = values[i];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}