#include "alphasparse/kernel.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < m; ++r)
        for (ALPHA_INT c = 0; c < n; c++)
        {
            ALPHA_Number tmp1, tmp2;
            alpha_mul(tmp1, y[index2(r, c, ldy)], beta);
            alpha_mul(tmp2, x[index2(r, c, ldx)], alpha);
            alpha_add(y[index2(r, c, ldy)], tmp1, tmp2);
        }

    num_threads = ((columns/512*24 + 24) < num_threads) ? (columns/512*24 + 24) : num_threads;

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT bcl = cross_block_low(tid, num_threads, columns);
        ALPHA_INT bch = cross_block_high(tid, num_threads, columns);
        ALPHA_INT bcs = cross_block_size(tid, num_threads, columns);

        for (ALPHA_INT ai = 0; ai < mat->nnz; ai++)
        {
            ALPHA_INT ac = mat->col_indx[ai];
            ALPHA_INT r = mat->row_indx[ai];
            if (ac < r)
            {
                ALPHA_Number val;
                alpha_mul(val, alpha, mat->values[ai]);
                ALPHA_Number val_c;
                alpha_conj(val_c,mat->values[ai]);
                alpha_mul(val_c, alpha, val_c);
                for (ALPHA_INT c = bcl; c < bch; ++c)
                    alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                for (ALPHA_INT c = bcl; c < bch; ++c)
                    alpha_madde(y[index2(ac, c, ldy)], val_c, x[index2(r, c, ldx)]);
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
