#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_CSR *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSR *transposed_mat;
    transpose_csr(mat, &transposed_mat);
    for (ALPHA_INT r = 0; r < transposed_mat->rows; ++r){
        for (ALPHA_INT i = transposed_mat->rows_start[r]; i < transposed_mat->rows_end[r]; ++i){
            ALPHA_INT c = transposed_mat->col_indx[i];
            if(r == c)
                transposed_mat->values[i].imag = 0.0 - transposed_mat->values[i].imag;
        }
    }
    alphasparse_status_t status = hermm_csr_n_hi_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csr(transposed_mat);
    return status;
}
