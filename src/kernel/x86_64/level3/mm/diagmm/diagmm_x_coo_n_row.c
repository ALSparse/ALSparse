#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT n = columns;
    ALPHA_INT _nnz = mat->nnz;    
    ALPHA_INT num_threads = alpha_get_thread_num();

    ALPHA_INT partition[num_threads + 1];
    partition[0] = 0;    

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT nnz = 0; nnz < _nnz; ++nnz)
    {
        ALPHA_INT tid = alpha_get_thread_id();
        if(mat->row_indx[nnz] != mat->row_indx[nnz+1])
        {
            partition[tid + 1] = nnz + 1;
            nnz = _nnz;
        }
    }
    
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT start = partition[tid];
        ALPHA_INT end = tid == num_threads - 1 ? _nnz : partition[tid + 1];
        ALPHA_INT or = mat->row_indx[start];
        for (ALPHA_INT nnz = start; nnz < end; ++nnz)
        {
            ALPHA_INT r = mat->row_indx[nnz];
            ALPHA_Number *Y = &y[index2(r, 0, ldy)];
            while(or <= r)
            {
                ALPHA_Number *TY = &y[index2(or, 0, ldy)];
                for (ALPHA_INT c = 0; c < n; c++)
                    alpha_mul(TY[c], TY[c], beta);

                or++;
            }

            if (mat->col_indx[nnz] == r)
            {
                ALPHA_Number val;
                alpha_mul(val, alpha, mat->values[nnz]);
                const ALPHA_Number *X = &x[index2(mat->col_indx[nnz], 0, ldx)];
                for (ALPHA_INT c = 0; c < n; ++c)
                    alpha_madde(Y[c], val, X[c]);
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
