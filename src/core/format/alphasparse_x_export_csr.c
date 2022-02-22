#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t ONAME(const alphasparse_matrix_t source,
                          alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT **rows_start,
                          ALPHA_INT **rows_end,
                          ALPHA_INT **col_indx,
                          ALPHA_Number **values)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_CSR, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    ALPHA_SPMAT_CSR *mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *rows_start = mat->rows_start;
    *rows_end = mat->rows_end;
    *col_indx = mat->col_indx;
    *values = mat->values;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}