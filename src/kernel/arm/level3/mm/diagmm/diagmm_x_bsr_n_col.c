#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_BSR *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT block_rowA = mat->rows;
    ALPHA_INT rowA = mat->rows * mat->block_size;
    ALPHA_Number diag[rowA]; 
    memset(diag, '\0', sizeof(ALPHA_Number) * rowA);
    ALPHA_INT bs = mat->block_size;
    ALPHA_INT num_threads = alpha_get_thread_num(); 
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif     
    for (ALPHA_INT ar = 0; ar < block_rowA; ++ar)
    {
        for (ALPHA_INT ai = mat->rows_start[ar]; ai < mat->rows_end[ar]; ++ai)
        {
            if (mat->col_indx[ai] == ar)
            {
                for(ALPHA_INT block_i = 0; block_i < bs; block_i++) 
                {
                    diag[ar*bs+block_i] = mat->values[ai*bs*bs + block_i*bs + block_i];
                }
            } 
        }   
    }
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif 
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            ALPHA_Number t1, t2;
            alpha_mul(t1, beta, y[index2(cc, cr, ldy)]);
            alpha_mul(t2, alpha, diag[cr]);
            alpha_mul(t2, t2, x[index2(cc, cr, ldx)]);
            alpha_add(y[index2(cc, cr, ldy)], t1, t2);
        }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
