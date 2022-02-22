#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    ALPHA_Number diag[m];
    memset(diag, '\0', m * sizeof(ALPHA_Number));
    int num_thread = alpha_get_thread_num(); 

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
        diag[r - 1] = A->values[indx];
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r <A->rows; r++)
        {
            ALPHA_Number temp;
            alpha_setzero(temp);
            ALPHA_INT start = A->pointers[r];
            ALPHA_INT end   = A->pointers[r + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end - 1; ++ai)
            {
                ALPHA_INT c = r - eles_num + idx;
                alpha_madde(temp, A->values[ai], y[out_y_col * ldy + c]);
                idx ++;
            }     

            ALPHA_Number t;
            alpha_mul(t, alpha, x[out_y_col * ldx + r]);
            alpha_sub(t, t, temp);
            alpha_div(y[out_y_col * ldy + r], t, diag[r]);
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
