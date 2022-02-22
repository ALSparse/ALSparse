#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_DIA *transposed_mat;
    transpose_dia(A, &transposed_mat);
    alphasparse_status_t status = trsv_dia_n_lo_plain(alpha, transposed_mat, x, y);
    destroy_dia(transposed_mat);
    return status;
}
