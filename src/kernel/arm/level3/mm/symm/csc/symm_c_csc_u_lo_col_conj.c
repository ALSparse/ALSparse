#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT cc=0; cc<columns; ++cc)
    {
        const ALPHA_Number *X = &x[index2(cc, 0, ldx)];
        ALPHA_Number *Y = &y[index2(cc, 0, ldy)];
        for (ALPHA_INT r = 0; r < mat->rows; r++)
        {
            ALPHA_Number tmp1, tmp2;
            alpha_mul(tmp1, X[r], alpha);
            alpha_mul(tmp2, Y[r], beta);
            alpha_add(Y[r], tmp1, tmp2); 
        }

        for (ALPHA_INT br = 0; br < mat->cols; ++br)
        {
            for (ALPHA_INT ai = mat->cols_start[br]; ai < mat->cols_end[br]; ++ai)
            {
                ALPHA_INT ar = mat->row_indx[ai];
                ALPHA_Number spval;
                alpha_conj(spval, mat->values[ai]);
                alpha_mule(spval, alpha);                
                if(ar > br) // non-diag element, deal with sym ele
                {
                    alpha_madde(Y[ar], spval, X[br]);
                    alpha_madde(Y[br], spval, X[ar]);
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}