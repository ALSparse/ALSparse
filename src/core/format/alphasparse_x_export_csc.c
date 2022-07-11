#include "alphasparse_cpu.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t ONAME(const alphasparse_matrix_t source,
                          alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT **cols_start,
                          ALPHA_INT **cols_end,
                          ALPHA_INT **row_indx,
                          ALPHA_Number **values)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_CSC, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    ALPHA_SPMAT_CSC *mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *cols_start = mat->cols_start;
    *cols_end = mat->cols_end;
    *row_indx = mat->row_indx;
    *values = mat->values;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
