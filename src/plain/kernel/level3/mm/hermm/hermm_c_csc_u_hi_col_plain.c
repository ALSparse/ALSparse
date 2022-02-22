#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT cc=0; cc<columns; ++cc)
    {
        const ALPHA_Complex *X = &x[index2(cc, 0, ldx)];
        ALPHA_Complex *Y = &y[index2(cc, 0, ldy)];
        for (ALPHA_INT r = 0; r < mat->rows; r++)
        {
            ALPHA_Complex tmp1, tmp2;
            alpha_mul(tmp1, X[r], alpha);
            alpha_mul(tmp2, Y[r], beta);
            alpha_add(Y[r], tmp1, tmp2); 
        }

        for (ALPHA_INT br = 0; br < mat->cols; ++br)
        {
            //double xval = X[br];
            for (ALPHA_INT ai = mat->cols_start[br]; ai < mat->cols_end[br]; ++ai)
            {
                // ai @ [ar, br] & [br, ar]
                ALPHA_INT ar = mat->row_indx[ai];
                ALPHA_Complex spval;
                ALPHA_Complex val_c;
                val_c.real = mat->values[ai].real;
                val_c.imag = 0.0 - mat->values[ai].imag;
                alpha_mul(val_c, alpha, val_c);
                //double spval = alpha * mat->values[ai];
                alpha_mul(spval, alpha, mat->values[ai]);                
                if(ar < br) // non-diag element, deal with sym ele
                {
                    //double symval = X[ar];
                    //Y[ar] += spval * xval;
                    alpha_madde(Y[ar], spval, X[br]);
                    //Y[br] += spval * symval;
                    alpha_madde(Y[br], val_c, X[ar]);
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}