#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_SKY *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    ALPHA_Complex diag[m];
    memset(diag, '\0', m * sizeof(ALPHA_Complex));
    ALPHA_INT num_thread = alpha_get_thread_num(); 
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT r = 0; r < m; r++)
    {
        const ALPHA_INT indx = A->pointers[r + 1] - 1;
		diag[r].real = A->values[indx].real;
        diag[r].imag = -A->values[indx].imag;
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
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
                    alpha_madde(temp, cv, y[ic * ldy + out_y_col]);
                }
            }

            ALPHA_Complex t;
            alpha_mul(t, alpha, x[c * ldx + out_y_col]);
            alpha_sub(t, t, temp);
            alpha_div(y[c * ldy + out_y_col], t, diag[c]);
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
