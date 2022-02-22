#ifdef _OPENMP 
#include<omp.h>
#endif 

#include "alphasparse/opt.h" 
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    ALPHA_SPMAT_COO *transposed_mat;
    transpose_coo(mat, &transposed_mat);
    int nnz = transposed_mat->nnz;
    alphasparse_status_t status = hermm_coo_n_hi_row(alpha,
                                                    transposed_mat,
                                                    x,
                                                    columns,
                                                    ldx,
                                                    beta,
                                                    y,
                                                    ldy);
    destroy_coo(transposed_mat);
    return status;

}