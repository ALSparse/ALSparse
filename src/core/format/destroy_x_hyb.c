
#include "alphasparse/format.h"
#include <alphasparse/util.h>

alphasparse_status_t ONAME(ALPHA_SPMAT_HYB *A)
{
    alpha_free(A->ell_val);
    alpha_free(A->ell_col_ind);
    alpha_free(A->coo_val);
    alpha_free(A->coo_row_val);
    alpha_free(A->coo_col_val);

    alpha_free_dcu(A->d_coo_col_val);
    alpha_free_dcu(A->d_coo_row_val);
    alpha_free_dcu(A->d_coo_val);
    alpha_free_dcu(A->d_ell_col_ind);
    alpha_free_dcu(A->d_ell_val);

    alpha_free(A);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
