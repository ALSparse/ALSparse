#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_SPMAT_DIA *conjugated_mat;
    transpose_conj_dia(A, &conjugated_mat);
    alphasparse_status_t status = trsv_dia_n_hi_plain(alpha, conjugated_mat, x, y);
    destroy_dia(conjugated_mat);
    return status;
}
