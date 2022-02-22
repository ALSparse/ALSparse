#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    for (ALPHA_INT c = A->cols - 1; c >= 0; c--)
    {
        ALPHA_Complex temp = {.real = 0.f, .imag = 0.f};
        for (ALPHA_INT ic = A->cols - 1; ic > c; ic--)
        {
            ALPHA_INT start = A->pointers[ic];
            ALPHA_INT end   = A->pointers[ic + 1];
            ALPHA_INT eles_num = ic - c;
            if(end - eles_num - 1 >= start)
            {
                ALPHA_Complex cv = A->values[end - eles_num - 1];
                alpha_conj(cv, cv);
                alpha_madde(temp, cv, y[ic]);
            }
                // temp += A->values[end - eles_num - 1] * y[ic];
        }
        
        ALPHA_Complex t = {.real = 0.f, .imag = 0.f};
        alpha_mul(t, alpha, x[c]);
        alpha_sub(y[c], t, temp);
        // y[c] = alpha * x[c] - temp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
