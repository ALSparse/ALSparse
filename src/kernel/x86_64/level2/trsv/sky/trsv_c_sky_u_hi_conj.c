#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
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
            alpha_madde(temp, cv, y[c]);
            idx ++;
        }     

        ALPHA_Complex t = {.real = 0.f, .imag = 0.f};
        alpha_mul(t, alpha, x[r]);
        alpha_sub(y[r], t, temp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS; 
}
