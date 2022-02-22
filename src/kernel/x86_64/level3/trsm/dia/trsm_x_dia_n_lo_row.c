#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_DIA *A, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    ALPHA_INT main_diag_pos = 0;
    ALPHA_Number diag[m];
    memset(diag, '\0', m * sizeof(ALPHA_Number));
    int num_thread = alpha_get_thread_num(); 

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT i = 0; i < A->ndiag; i++)
    {
        if(A->distance[i] == 0)
        {
            main_diag_pos = i;
            for (ALPHA_INT r = 0; r < A->rows; r++)
            {
                diag[r] = A->values[i * A->lval + r];
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r < m; r++)
        {
            ALPHA_Number temp;
            alpha_setzero(temp);
            for (ALPHA_INT ndiag = 0; ndiag < main_diag_pos; ndiag++)
            {
                if (-A->distance[ndiag] <= r)
                {
                    ALPHA_INT ac = r + A->distance[ndiag];
                    alpha_madde(temp, A->values[ndiag * A->lval + r], y[ac * ldy + out_y_col]);
                }
            }
            ALPHA_Number t;
            alpha_setzero(t);
            alpha_mul(t, alpha, x[r * ldx + out_y_col]);
            alpha_sub(t, t, temp);
            alpha_div(y[r * ldy + out_y_col], t, diag[r]);
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
