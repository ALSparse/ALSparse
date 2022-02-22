
#include "alphasparse/format.h"
#include <alphasparse/util.h>

alphasparse_status_t ONAME(ALPHA_SPMAT_GEBSR *A)
{
    alpha_free(A->values);
    alpha_free(A->rows_start);
    alpha_free(A->col_indx);
    
    alpha_free_dcu(A->d_col_indx);
    alpha_free_dcu(A->d_rows_ptr);
    alpha_free_dcu(A->d_values);

    alpha_free(A);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
