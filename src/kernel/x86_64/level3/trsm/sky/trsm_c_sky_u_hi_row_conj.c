#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_thread = alpha_get_thread_num(); 
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r <A->rows; r++)
        {
            ALPHA_Complex temp = {.real = 0.f, .imag = 0.f};

            ALPHA_INT start = A->pointers[r];
            ALPHA_INT end   = A->pointers[r + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end - 1; ++ai)
            {
                ALPHA_INT c = r - eles_num + idx;
                ALPHA_Complex cv = A->values[ai];
                alpha_conj(cv, cv);
                alpha_madde(temp, cv, y[c * ldy + out_y_col]);
                idx ++;
            }     

            ALPHA_Complex t;
            alpha_mul(t, alpha, x[r * ldx + out_y_col]);
            alpha_sub(y[r * ldy + out_y_col], t, temp);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
