#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        const ALPHA_Number *X = &x[index2(cc, 0, ldx)];
        ALPHA_Number *Y = &y[index2(cc, 0, ldy)];
        for(ALPHA_INT i=0; i < mat->rows; ++i)
        {
            ALPHA_Number tmp1, tmp2;
            alpha_mul(tmp1, alpha, X[i]);
            alpha_mul(tmp2, beta, Y[i]);
            alpha_add(Y[i], tmp1, tmp2);//y[index2(ix,iy,ldy)] *= beta;
        }

        for(ALPHA_INT br = 0; br < mat->cols; ++br)
        {
            ALPHA_Number xval = X[br];
            for (ALPHA_INT ai = mat->cols_start[br]; ai < mat->cols_end[br]; ++ai)
            {
                ALPHA_INT ar = mat->row_indx[ai];
                if(br > ar)
                {
                    ALPHA_Number spval;// = alpha * mat->values[ai];
                    alpha_mul(spval, alpha, mat->values[ai]);
                    alpha_madde(Y[ar], spval, xval);
                    //Y[ar] += spval * xval; 
                }                
            }
        }

    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}