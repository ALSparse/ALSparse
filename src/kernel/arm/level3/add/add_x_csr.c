#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "memory.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *A, const ALPHA_Number alpha, const ALPHA_SPMAT_CSR *B, ALPHA_SPMAT_CSR **matC)
{
    ALPHA_SPMAT_CSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSR));
    *matC = mat;
    ALPHA_INT rowA = A->rows;
    ALPHA_INT rowB = B->rows;
    ALPHA_INT colA = A->cols;
    ALPHA_INT colB = B->cols;

    check_return(rowA != rowB, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(colA != colB, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    ALPHA_INT *rows_offset = alpha_memalign((rowA + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->rows = rowA;
    mat->cols = colB;
    mat->rows_start = rows_offset;
    mat->rows_end = rows_offset + 1;
    ALPHA_INT num_threads = alpha_get_thread_num();

    ALPHA_INT count = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) reduction(+ \
                                                            : count)
#endif
    for (ALPHA_INT r = 0; r < rowA; ++r)
    {
        ALPHA_INT ai = A->rows_start[r];
        ALPHA_INT rea = A->rows_end[r];
        ALPHA_INT bi = B->rows_start[r];
        ALPHA_INT reb = B->rows_end[r];
        while (ai < rea && bi < reb)
        {
            ALPHA_INT cb = B->col_indx[bi];
            ALPHA_INT ca = A->col_indx[ai];
            if (ca < cb)
            {
                ai += 1;
            }
            else if (cb < ca)
            {
                bi += 1;
            }
            else
            {
                ai += 1;
                bi += 1;
            }
            count += 1;
        }
        if (ai == rea)
        {
            count += reb - bi;
        }
        else
        {
            count += rea - ai;
        }
    }
    mat->col_indx = alpha_memalign(count * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(count * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);

    // add
    size_t index = 0;
    mat->rows_start[0] = 0;

    for (ALPHA_INT r = 0; r < rowA; ++r)
    {
        ALPHA_INT ai = A->rows_start[r];
        ALPHA_INT rea = A->rows_end[r];
        ALPHA_INT bi = B->rows_start[r];
        ALPHA_INT reb = B->rows_end[r];
        while (ai < rea && bi < reb)
        {
            ALPHA_INT ca = A->col_indx[ai];
            ALPHA_INT cb = B->col_indx[bi];
            if (ca < cb)
            {
                mat->col_indx[index] = ca;
                alpha_mul(mat->values[index], A->values[ai], alpha);
                ai += 1;
            }
            else if (cb < ca)
            {
                mat->col_indx[index] = cb;
                mat->values[index] = B->values[bi];
                bi += 1;
            }
            else
            {
                mat->col_indx[index] = ca;
                alpha_madd(mat->values[index], A->values[ai], alpha, B->values[bi]);
                ai += 1;
                bi += 1;
            }
            index += 1;
        }
        if (ai == rea)
        {
            for (; bi < reb; ++bi, ++index)
            {
                mat->col_indx[index] = B->col_indx[bi];
                mat->values[index] = B->values[bi];
            }
        }
        else
        {
            for (; ai < rea; ++ai, ++index)
            {
                mat->col_indx[index] = A->col_indx[ai];
                alpha_mul(mat->values[index], A->values[ai], alpha);
            }
        }
        mat->rows_end[r] = index;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
