#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t ONAME(const alphasparse_matrix_t source,
                          alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT *width,
                          ALPHA_INT **col_indx,
                          ALPHA_Number **values)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_ELL, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    ALPHA_SPMAT_ELL *mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *width = mat->ld;
    *col_indx = mat->indices;
    *values = mat->values;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}