
#include "alphasparse/format.h"
#include <alphasparse/util.h>

alphasparse_status_t ONAME(ALPHA_SPMAT_SKY *A)
{
    alpha_free(A->pointers);
    alpha_free(A->values);

    alpha_free(A);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
