#include "alphasparse/kernel_plain.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_BSR *A, const ALPHA_SPMAT_BSR *B, ALPHA_SPMAT_BSR **matC)
{
    ALPHA_SPMAT_BSR *transposed_mat;
    transpose_bsr(A, &transposed_mat); //将mat转置
    alphasparse_status_t status = spmm_bsr_plain(transposed_mat, B, matC); //再调用乘法
    destroy_bsr(transposed_mat);
    return status;
}
