#include "alphasparse_cpu.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

alphasparse_status_t ONAME(alphasparse_matrix_t *A,
                          const alphasparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                          const ALPHA_INT rows,
                          const ALPHA_INT cols,
                          ALPHA_INT *cols_start,
                          ALPHA_INT *cols_end,
                          ALPHA_INT *row_indx,
                          ALPHA_Number *values)
{
    alphasparse_matrix *AA = alpha_malloc(sizeof(alphasparse_matrix));
    *A = AA;
    ALPHA_SPMAT_CSC *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSC));
    AA->format = ALPHA_SPARSE_FORMAT_CSC;
    AA->datatype = ALPHA_SPARSE_DATATYPE;
    AA->mat = mat;
    ALPHA_INT nnz = cols_end[cols - 1];
    mat->rows = rows;
    mat->cols = cols;
    ALPHA_INT *cols_offset = alpha_memalign((cols + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->row_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    mat->cols_start = cols_offset;
    mat->cols_end = cols_offset + 1;
    if (indexing == ALPHA_SPARSE_INDEX_BASE_ZERO)
    {
        cols_offset[0] = cols_start[0];
        for (ALPHA_INT i = 0; i < rows; i++)
        {
            mat->cols_end[i] = cols_end[i];
        }
        for (ALPHA_INT i = 0; i < nnz; i++)
        {
            mat->row_indx[i] = row_indx[i];
            mat->values[i] = values[i];
        }
    }
    else
    {
        cols_offset[0] = cols_start[0] - 1;
        for (ALPHA_INT i = 0; i < rows; i++)
        {
            mat->cols_end[i] = cols_end[i] - 1;
        }
        for (ALPHA_INT i = 0; i < nnz; i++)
        {
            mat->row_indx[i] = row_indx[i] - 1;
            mat->values[i] = values[i];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
