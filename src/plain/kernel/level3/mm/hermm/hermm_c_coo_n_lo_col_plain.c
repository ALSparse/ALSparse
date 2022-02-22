/** 
 * @Author: Zjj
 * @Date: 2020-06-25 11:21:20
 * @LastEditors: Zjj
 * @LastEditTime: 2020-06-25 17:34:49
 */
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT j = 0; j < columns; j++)
        for (ALPHA_INT i = 0; i < mat->rows; i++)
            alpha_mul(y[i + j * ldy], y[i + j * ldy], beta);
        // y[i] *= beta;
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT nnz = 0; nnz < mat->nnz; ++nnz)
        {
            ALPHA_INT cr = mat->row_indx[nnz];
            ALPHA_INT ac = mat->col_indx[nnz];
            // printf("get A (%d,%d):(%.10f,%.10f)\n",cr,ac,mat->values[nnz].real,mat->values[nnz].imag);

            if (ac < cr)
            {
                ALPHA_Number t,t_conj;
                ALPHA_Number origin_val = mat->values[nnz];
                ALPHA_Number conj_val;
                alpha_conj(conj_val,origin_val);
                
                alpha_mul(t,origin_val, alpha);
                alpha_mul(t_conj,conj_val, alpha);

                alpha_madde(y[index2(cc, cr, ldy)] , t, x[index2(cc, ac, ldx)]);
                alpha_madde(y[index2(cc, ac,  ldy)], t_conj, x[index2(cc, cr, ldx)]);
                // y[index2(cc, cr, ldy)] += alpha * mat->values[nnz] * x[index2(cc, ac, ldx)];
                // y[index2(cc, ac, ldy)] += alpha * mat->values[nnz] * x[index2(cc, cr, ldx)];
            }
            else if (ac == cr)
            {
                ALPHA_Number t;
                alpha_setzero(t);
                alpha_mul(t, mat->values[nnz], alpha);
                // printf("before multiplying x , y (%d,%d):(%.10f,%.10f)\n",cr,cc,t.real,t.imag);
                alpha_madde(y[index2(cc, cr, ldy)], t, x[index2(cc, ac, ldx)]);
                // printf("multiplying x , x (%d,%d):(%.10f,%.10f)\n",ac,cc,x[index2(cc, ac, ldx)].real,x[index2(cc, ac, ldx)].imag);
                // printf("after multiplying x , y (%d,%d):(%.10f,%.10f)\n",cr,cc,y[index2(cc, cr, ldy)].real,y[index2(cc, cr, ldy)].imag);

                // y[index2(cc, cr, ldy)] += alpha * mat->values[nnz] * x[index2(cc, ac, ldx)];
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
