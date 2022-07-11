#include "alphasparse_cpu.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t ONAME(const alphasparse_matrix_t source,
                          alphasparse_index_base_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT *nnz,
                          ALPHA_INT *ell_width,
                          ALPHA_Number **ell_val,
                          ALPHA_INT **ell_col_ind,
                          ALPHA_Number **coo_val,
                          ALPHA_INT **coo_row_val,
                          ALPHA_INT **coo_col_val)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_HYB, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    ALPHA_SPMAT_HYB *mat = source->mat;
    
    *indexing    = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows        = mat->rows;
    *cols        = mat->cols;
    *nnz         = mat->nnz;
    *ell_width   = mat->ell_width;
    *ell_val     = mat->ell_val;
    *ell_col_ind = mat->ell_col_ind;
    *coo_val     = mat->coo_val;
    *coo_row_val = mat->coo_row_val;
    *coo_col_val = mat->coo_col_val;

     return ALPHA_SPARSE_STATUS_SUCCESS;
}