#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <stdbool.h>
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *A, const ALPHA_SPMAT_CSR *B, ALPHA_SPMAT_CSR **matC)
{
    check_return(B->cols != A->rows, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    ALPHA_SPMAT_CSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSR));
    *matC = mat;
    mat->rows = A->rows;
    mat->cols = B->cols;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    // 计算所需空间
    bool *flag = alpha_memalign(sizeof(bool) * n, DEFAULT_ALIGNMENT);
    ALPHA_INT nnz = 0;
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        memset(flag, '\0', sizeof(bool) * n);
        for (ALPHA_INT ai = A->rows_start[ar]; ai < A->rows_end[ar]; ai++)
        {
            ALPHA_INT br = A->col_indx[ai];
            for (ALPHA_INT bi = B->rows_start[br]; bi < B->rows_end[br]; bi++)
            {
                if (!flag[B->col_indx[bi]])
                {
                    nnz += 1;
                    flag[B->col_indx[bi]] = true;
                }
            }
        }
    }
    alpha_free(flag);

    ALPHA_INT *row_offset = alpha_memalign(sizeof(ALPHA_INT) * (m + 1), DEFAULT_ALIGNMENT);
    mat->rows_start = row_offset;
    mat->rows_end = row_offset + 1;
    mat->col_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    memset(mat->values, '\0', sizeof(ALPHA_Number) * nnz);

    ALPHA_Number *values = alpha_memalign(sizeof(ALPHA_Number) * n, DEFAULT_ALIGNMENT);
    bool *write_back = alpha_memalign(sizeof(bool) * n, DEFAULT_ALIGNMENT);

    ALPHA_INT index = 0;
    mat->rows_start[0] = 0;
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        memset(values, '\0', sizeof(ALPHA_Number) * n);
        memset(write_back, '\0', sizeof(bool) * n);
        for (ALPHA_INT ai = A->rows_start[ar]; ai < A->rows_end[ar]; ai++)
        {
            ALPHA_INT br = A->col_indx[ai];
            ALPHA_Number av = A->values[ai];
            for (ALPHA_INT bi = B->rows_start[br]; bi < B->rows_end[br]; bi++)
            {
                ALPHA_INT bc = B->col_indx[bi];
                write_back[bc] = true;
                alpha_madde(values[bc], av, B->values[bi]);
            }
        }
        for (ALPHA_INT c = 0; c < n; c++)
        {
            if (write_back[c])
            {
                mat->col_indx[index] = c;
                mat->values[index] = values[c];
                index += 1;
            }
        }
        mat->rows_end[ar] = index;
    }

    alpha_free(values);
    alpha_free(write_back);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
