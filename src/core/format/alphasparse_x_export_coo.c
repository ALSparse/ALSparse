#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t ONAME(const alphasparse_matrix_t source,
                          alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT **row_indx,
                          ALPHA_INT **col_indx,
                          ALPHA_Number **values,
                          ALPHA_INT *nnz)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_COO, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    ALPHA_SPMAT_COO *mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *row_indx = mat->row_indx;
    *col_indx = mat->col_indx;
    *values = mat->values;
    *nnz = mat->nnz;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
