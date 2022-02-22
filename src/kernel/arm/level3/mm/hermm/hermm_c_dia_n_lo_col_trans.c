#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>
#include <stdlib.h>

alphasparse_status_t ONAME(const ALPHA_Complex alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Complex *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex beta, ALPHA_Complex *y, const ALPHA_INT ldy)
{    
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif     
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        ALPHA_Complex* Y = &y[index2(cc,0,ldy)];
        for (ALPHA_INT i = 0; i < mat->rows; i++)
            alpha_mul(Y[i],Y[i],beta);
        const ALPHA_Complex* X = &x[index2(cc,0,ldx)];
        for(ALPHA_INT di = 0; di < mat->ndiag;++di){
            ALPHA_INT d = mat->distance[di];
            if(d < 0){
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
            if(d == 0){
                for(ALPHA_INT r = 0; r < mat->rows; ++r){
                    ALPHA_Complex val;
                    alpha_mul_2c(val,mat->values[index2(di,r,mat->lval)],alpha);
                    alpha_madde(Y[r],val,X[r]);
                }
            }
        } 	
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
