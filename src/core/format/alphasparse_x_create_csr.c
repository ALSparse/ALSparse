#include "alphasparse.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

alphasparse_status_t ONAME(alphasparse_matrix_t *A,
                          const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                          const ALPHA_INT rows,
                          const ALPHA_INT cols,
                          ALPHA_INT *rows_start,
                          ALPHA_INT *rows_end,
                          ALPHA_INT *col_indx,
                          ALPHA_Number *values)
{
    alphasparse_matrix *AA = alpha_malloc(sizeof(alphasparse_matrix));
    *A = AA;
    ALPHA_SPMAT_CSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSR));
    AA->format = ALPHA_SPARSE_FORMAT_CSR;
    AA->datatype = ALPHA_SPARSE_DATATYPE;
    AA->mat = mat;
    ALPHA_INT nnz = rows_end[rows - 1] - rows_start[0];
    mat->rows = rows;
    mat->cols = cols;
    ALPHA_INT *rows_offset = alpha_memalign((rows + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->col_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    mat->rows_start = rows_offset;
    mat->rows_end = rows_offset + 1;
    if (indexing == ALPHA_SPARSE_INDEX_BASE_ZERO)
    {
        mat->rows_start[0] = rows_start[0];
        for (ALPHA_INT i = 0; i < rows; i++)
        {
            mat->rows_end[i] = rows_end[i];
        }
        for (ALPHA_INT i = 0; i < nnz; i++)
        {
            mat->col_indx[i] = col_indx[i];
            mat->values[i] = values[i];
        }
    }
    else
    {
        mat->rows_start[0] = rows_start[0] - 1;
        for (ALPHA_INT i = 0; i < rows; i++)
        {
            mat->rows_end[i] = rows_end[i] - 1;
        }
        for (ALPHA_INT i = 0; i < nnz; i++)
        {
            mat->col_indx[i] = col_indx[i] - 1;
            mat->values[i] = values[i];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}