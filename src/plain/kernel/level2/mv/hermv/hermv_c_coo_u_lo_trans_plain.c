/** 
 * @Author: Zjj
 * @Date: 2020-06-24 09:32:52
 * @LastEditors: Zjj
 * @LastEditTime: 2020-06-24 14:26:03
 */
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <stdio.h>


alphasparse_status_t
ONAME(const ALPHA_Complex alpha,
      const ALPHA_SPMAT_COO *A,
      const ALPHA_Complex *x,
      const ALPHA_Complex beta,
      ALPHA_Complex *y)
{
#ifdef PRINT
	printf("kernel hermv_coo_u_lo_trans_plain called\n");
#endif
    //TODO 
    ALPHA_SPMAT_COO *transposed_mat;
    transpose_coo(A, &transposed_mat);
    alphasparse_status_t status = hermv_coo_u_hi_plain(    alpha,
		                                                    transposed_mat,
                                                            x,
                                                            beta,
                                                            y);
    destroy_coo(transposed_mat);
    return status;
}