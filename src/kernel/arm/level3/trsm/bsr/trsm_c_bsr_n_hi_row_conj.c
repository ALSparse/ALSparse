#include "alphasparse/opt.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    const ALPHA_INT num_thread = alpha_get_thread_num(); 
    ALPHA_SPMAT_BSR *conjugated_mat;
    transpose_conj_bsr(A, &conjugated_mat);
    alphasparse_status_t status = trsm_bsr_n_lo_row(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_bsr(conjugated_mat);
    return status;
}
