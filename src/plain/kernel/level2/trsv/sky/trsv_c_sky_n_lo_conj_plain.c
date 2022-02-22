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
        }

        ALPHA_Complex t = {.real = 0.f, .imag = 0.f};
        alpha_mul(t, alpha, x[c]);
        alpha_sub(t, t, temp);
        alpha_div(y[c], t, diag[c]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
