
#include "alphasparse/format.h"
#include <alphasparse/util.h>

alphasparse_status_t ONAME(ALPHA_SPMAT_ELL *A)
{
    alpha_free(A->values);
    alpha_free(A->indices);

    alpha_free_dcu(A->d_indices);
    alpha_free_dcu(A->d_values);

    alpha_free(A);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
