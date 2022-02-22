#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    ALPHA_Complex diag[A->cols];
    memset(diag, '\0', A->rows * sizeof(ALPHA_Complex));
    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
		diag[r - 1].real = A->values[indx].real;
        diag[r - 1].imag = -A->values[indx].imag;
    }

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
            alpha_madde(temp, cv, y[c]);
            idx ++;
        }     

        ALPHA_Complex t = {.real = 0.f, .imag = 0.f};
        alpha_mul(t, alpha, x[r]);
        alpha_sub(t, t, temp);
        alpha_div(y[r], t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
