/** 
 * @Author: Zjj
 * @Date: 2020-06-25 11:21:20
 * @LastEditors: Zjj
 * @LastEditTime: 2020-06-25 14:31:29
 */
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
#ifdef PRINT
	printf("kernel hermm_coo_n_hi_row_trans_plain called\n");
#endif
    //TODO faster without invoking transposed 
    ALPHA_SPMAT_COO *transposed_mat;
    transpose_coo(mat, &transposed_mat);
    int nnz = transposed_mat->nnz;

    //TODO 
    // for(int i = 0 ; i < nnz ; i++){
    //     int r  = transposed_mat->row_indx[i];
    //     int c  = transposed_mat->col_indx[i];
    //     if(r==c){
    //         transposed_mat->values[i].imag = - transposed_mat->values[i].imag;
    //     }

    // }
    alphasparse_status_t status = hermm_coo_n_lo_row_plain(alpha,
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