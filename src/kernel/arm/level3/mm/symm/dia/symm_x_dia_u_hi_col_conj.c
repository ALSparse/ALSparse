#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
#ifdef COMPLEX
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(ALPHA_INT j = 0; j < columns; j++)
        for (ALPHA_INT i = 0; i < mat->rows; i++){
            alpha_mul(y[index2(j,i,ldy)],y[index2(j,i,ldy)],beta);
            alpha_madde(y[index2(j,i,ldy)],x[index2(j,i,ldx)],alpha);
        }
      
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif      
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        ALPHA_Number* Y = &y[index2(cc,0,ldy)];
        const ALPHA_Number* X = &x[index2(cc,0,ldx)];
        for(ALPHA_INT di = 0; di < mat->ndiag;++di){
            ALPHA_INT d = mat->distance[di];
            if(d > 0){
                ALPHA_INT ars = alpha_max(0,-d);
                ALPHA_INT acs = alpha_max(0,d);
                ALPHA_INT an = alpha_min(mat->rows - ars,mat->cols - acs);
                for(ALPHA_INT i = 0; i < an; ++i){
                    ALPHA_INT ar = ars + i;
                    ALPHA_INT ac = acs + i;
                    ALPHA_Number val;
                    alpha_mul_2c(val,mat->values[index2(di,ar,mat->lval)],alpha);
                    alpha_madde(Y[ar],val,X[ac]);
                    alpha_madde(Y[ac],val,X[ar]);
                }
            }
        } 	
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
#else
	return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif
}
