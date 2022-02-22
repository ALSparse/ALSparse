#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT j = 0; j < columns; j++)
        for (ALPHA_INT i = 0; i < mat->rows; i++){
            alpha_mul(y[index2(j,i,ldy)],y[index2(j,i,ldy)],beta);
            alpha_madde(y[index2(j,i,ldy)],x[index2(j,i,ldx)],alpha);
        }
            
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        ALPHA_Complex* Y = &y[index2(cc,0,ldy)];
        const ALPHA_Complex* X = &x[index2(cc,0,ldx)];
        for(ALPHA_INT di = 0; di < mat->ndiag;++di){
            ALPHA_INT d = mat->distance[di];
            if(d > 0){
                ALPHA_INT ars = alpha_max(0,-d);
                ALPHA_INT acs = alpha_max(0,d);
                ALPHA_INT an = alpha_min(mat->rows - ars,mat->cols - acs);
                for(ALPHA_INT i = 0; i < an; ++i){
                    ALPHA_INT ar = ars + i;
                    ALPHA_INT ac = acs + i;
                    ALPHA_Complex val,val_c;
                    alpha_mul(val,mat->values[index2(di,ar,mat->lval)],alpha);
                    alpha_mul_2c(val_c,mat->values[index2(di,ar,mat->lval)],alpha);

                    alpha_madde(Y[ar],val_c,X[ac]);
                    alpha_madde(Y[ac],val,X[ar]);
                }
            }
        } 	
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
