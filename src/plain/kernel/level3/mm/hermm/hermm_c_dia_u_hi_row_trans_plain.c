#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT r = 0; r < mat->rows; r++)
        for(ALPHA_INT c = 0; c < columns; c++){
            alpha_mul(y[index2(r,c,ldy)],y[index2(r,c,ldy)],beta);
            alpha_madde(y[index2(r,c,ldy)],x[index2(r,c,ldx)],alpha);
        }
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

                for(ALPHA_INT bc = 0;bc < columns;++bc){
                    alpha_madde(y[index2(ar,bc,ldy)],val_c,x[index2(ac,bc,ldx)]);
                    alpha_madde(y[index2(ac,bc,ldy)],val,x[index2(ar,bc,ldx)]);
                }
            }
        }
    } 	
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
