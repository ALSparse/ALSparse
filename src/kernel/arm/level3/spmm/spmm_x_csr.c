#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>

#ifdef _OPENMP
#include <omp.h>
#endif

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *A, const ALPHA_SPMAT_CSR *B, ALPHA_SPMAT_CSR **matC)
{
    check_return(B->cols != A->rows, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    ALPHA_SPMAT_CSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSR));
    *matC = mat;
    mat->rows = A->rows;
    mat->cols = B->cols;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    ALPHA_INT64 flop[m];
    memset(flop,'\0',m*sizeof(ALPHA_INT64));
    ALPHA_INT *row_offset = alpha_memalign(sizeof(ALPHA_INT) * (m + 1), DEFAULT_ALIGNMENT);
    mat->rows_start = row_offset;
    mat->rows_end = row_offset + 1;
    memset(row_offset,'\0',sizeof(ALPHA_INT)*(m+1));

    ALPHA_INT num_thread = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        bool flag[n];
        memset(flag, '\0', sizeof(bool) * n);
        for (ALPHA_INT ai = A->rows_start[ar]; ai < A->rows_end[ar]; ai++)
        {
            ALPHA_INT br = A->col_indx[ai];
            flop[ar] += B->rows_end[br] - B->rows_start[br];
            for (ALPHA_INT bi = B->rows_start[br]; bi < B->rows_end[br]; bi++)
            {
                if (!flag[B->col_indx[bi]])
                {
                    mat->rows_end[ar] += 1;
                    flag[B->col_indx[bi]] = true;
                }
            }
        }
    }
    
    for(ALPHA_INT i = 1;i < m;++i)
    {
        flop[i] += flop[i - 1];
        mat->rows_end[i] += mat->rows_end[i-1];
    }
    ALPHA_INT nnz = mat->rows_end[m-1];

    mat->col_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        ALPHA_Number values[n];
        memset(values, '\0', sizeof(ALPHA_Number) * n);
        bool write_back[n];
        memset(write_back, '\0', sizeof(bool) * n);
        for (ALPHA_INT ai = A->rows_start[ar]; ai < A->rows_end[ar]; ai++)
        {
            ALPHA_INT br = A->col_indx[ai];
            ALPHA_Number av = A->values[ai];
            ALPHA_INT bis = B->rows_start[br];
            ALPHA_INT bie = B->rows_end[br];
            ALPHA_INT bil = bie-bis;
            const ALPHA_INT* B_col = &B->col_indx[bis];
            const ALPHA_Number* B_val = &B->values[bis];
            ALPHA_INT bi = 0;
            for (; bi < bil-3; bi+=4)
            {
                ALPHA_INT bc0 = B_col[bi];
                ALPHA_INT bc1 = B_col[bi+1];
                ALPHA_INT bc2 = B_col[bi+2];
                ALPHA_INT bc3 = B_col[bi+3];
                ALPHA_Number bv0 = B_val[bi];
                ALPHA_Number bv1 = B_val[bi+1];
                ALPHA_Number bv2 = B_val[bi+2];
                ALPHA_Number bv3 = B_val[bi+3];
                alpha_madde(values[bc0], av, bv0);
                alpha_madde(values[bc1], av, bv1);
                alpha_madde(values[bc2], av, bv2);
                alpha_madde(values[bc3], av, bv3);
                write_back[bc0] = true;
                write_back[bc1] = true;
                write_back[bc2] = true;
                write_back[bc3] = true;
            }
            for (; bi < bil; bi++)
            {
                ALPHA_INT bc = B_col[bi];
                ALPHA_Number bv = B_val[bi];
                alpha_madde(values[bc], av, bv);
                write_back[bc] = true;
            }
        }
        
        ALPHA_INT index = mat->rows_start[ar];
        for (ALPHA_INT c = 0; c < n; c++)
        {
            if (write_back[c])
            {
                mat->col_indx[index] = c;
                mat->values[index] = values[c];
                index += 1;
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
