#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_CSC *transposed_mat;
    transpose_csc(mat, &transposed_mat);
    for (ALPHA_INT c = 0; c < transposed_mat->cols; ++c){
        for (ALPHA_INT i = transposed_mat->cols_start[c]; i < transposed_mat->cols_end[c]; ++i){
            ALPHA_INT r = transposed_mat->row_indx[i];
            if(r == c)
                transposed_mat->values[i].imag = 0.0 - transposed_mat->values[i].imag;
        }
    }
    alphasparse_status_t status = hermm_csc_n_lo_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csc(transposed_mat);
    return status;
}