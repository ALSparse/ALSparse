#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < mat->rows; r++)
        for(ALPHA_INT c = 0; c < columns; c++)
            alpha_mul(y[index2(r,c,ldy)],y[index2(r,c,ldy)],beta);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT bcl = cross_block_low(tid,num_threads,columns);
        ALPHA_INT bch = cross_block_high(tid,num_threads,columns);
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
                    alpha_mul(val,mat->values[index2(di,ar,mat->lval)],alpha);
                    for(ALPHA_INT bc = bcl;bc < bch;++bc){
                        alpha_madde(y[index2(ar,bc,ldy)],val,x[index2(ac,bc,ldx)]);
                        alpha_madde(y[index2(ac,bc,ldy)],val,x[index2(ar,bc,ldx)]);
                    }
                }
            }
            if(d == 0){
                for(ALPHA_INT r = 0; r < mat->rows; ++r){
                    ALPHA_Number val;
                    alpha_mul(val,mat->values[index2(di,r,mat->lval)],alpha);
                    for(ALPHA_INT bc = bcl;bc < bch;++bc){
                        alpha_madde(y[index2(r,bc,ldy)],val,x[index2(r,bc,ldx)]);
                    }
                }
            }
        } 	
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
