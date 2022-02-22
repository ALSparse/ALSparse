#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <stdbool.h>
#include <memory.h>
#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *A, const ALPHA_SPMAT_CSC *B, ALPHA_SPMAT_CSC **matC)
{
    check_return(A->cols != B->rows, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    ALPHA_SPMAT_CSC *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSC));
    *matC = mat;
    mat->rows = A->rows;
    mat->cols = B->cols;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    bool *flag = alpha_memalign(sizeof(bool) * m, DEFAULT_ALIGNMENT);
    ALPHA_INT nnz = 0;
    ALPHA_INT num_thread = alpha_get_thread_num();
    for (ALPHA_INT bc = 0; bc < n; bc++)
    {
        memset(flag, '\0', sizeof(bool) * m);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread) reduction(+:nnz)
#endif
        for (ALPHA_INT bi = B->cols_start[bc]; bi < B->cols_end[bc]; bi++)
        {
            ALPHA_INT ac = B->row_indx[bi];
            for (ALPHA_INT ai = A->cols_start[ac]; ai < A->cols_end[ac]; ai++)
            {
                if (!flag[A->row_indx[ai]])
                {
                    nnz += 1;
                    flag[A->row_indx[ai]] = true;
                }
            }
        }
    }
    alpha_free(flag);

    ALPHA_INT *col_offset = alpha_memalign(sizeof(ALPHA_INT) * (n + 1), DEFAULT_ALIGNMENT);
    mat->cols_start = col_offset;
    mat->cols_end = col_offset + 1;
    mat->row_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);

    ALPHA_Number *values = alpha_memalign(sizeof(ALPHA_Number) * m, DEFAULT_ALIGNMENT);

    ALPHA_INT index = 0;
    mat->cols_start[0] = 0;
    for (ALPHA_INT bc = 0; bc < n; bc++) 
    {
        memset(values, '\0', sizeof(ALPHA_Number) * m); 
        bool *flagg = alpha_memalign(sizeof(bool) * m, DEFAULT_ALIGNMENT);
        memset(flagg, '\0', sizeof(bool) * m);
        for (ALPHA_INT bi = B->cols_start[bc]; bi < B->cols_end[bc]; bi++) 
        {
            ALPHA_INT ac = B->row_indx[bi];
            ALPHA_Number bv = B->values[bi];
            bv = B->values[bi];
            for (ALPHA_INT ai = A->cols_start[ac]; ai < A->cols_end[ac]; ai++)
            {
                ALPHA_INT ar = A->row_indx[ai];
                ALPHA_Number tmp;
                alpha_mul(tmp, bv, A->values[ai]);
                alpha_adde(values[ar], tmp);
                flagg[ar] = true;
            }
        }
        for (ALPHA_INT r = 0; r < m; r++)
        {
            if(flagg[r])
            {
                mat->row_indx[index] = r;
                mat->values[index] = values[r];
                index += 1;
            }
        }
        mat->cols_end[bc] = index;
        alpha_free(flagg);
    }

    alpha_free(values);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
