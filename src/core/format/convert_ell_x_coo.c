#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <stdio.h>

static void print_coo_s(const spmat_coo_s_t *mat)
{
    printf("nnz:%d, cols:%d, rows:%d\n", mat->nnz, mat->cols, mat->rows);
    for (ALPHA_INT i = 0; i < mat->nnz; i++)
    {
        printf("#%d, val:%f, row:%d, col:%d\n", i, mat->values[i], mat->row_indx[i], mat->col_indx[i]);
    }
    printf("=====================================\n\n");
}

static void print_ell_s(const spmat_ell_s_t *mat)
{
    printf("ld:%d, cols:%d, rows:%d\n", mat->ld, mat->cols, mat->rows);
    for(ALPHA_INT i = 0; i < mat->ld; i++)
    {
        for(ALPHA_INT j = 0; j < mat->rows; j++)
        {
            printf("%f ", mat->values[i*mat->rows + j]);
        }
        printf("\n");
    }
    printf("=====================================\n\n");
}

alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_ELL **dest)
{
    ALPHA_SPMAT_ELL *mat = alpha_malloc(sizeof(ALPHA_SPMAT_ELL));
    *dest = mat;

    ALPHA_SPMAT_CSR *csr;
    convert_csr_coo(source, &csr);

    ALPHA_INT m = csr->rows;
    ALPHA_INT n = csr->cols;

    mat->rows = m;
    mat->cols = n;

    ALPHA_INT ld = 0;
    for (ALPHA_INT i = 0; i < m; i++)
    {
        ALPHA_INT row_nnz = csr->rows_end[i] - csr->rows_start[i];
        ld = ld > row_nnz ? ld : row_nnz;
    }
    mat->ld = ld;
    if((uint64_t )ld * m >= 1l<<31){
        fprintf(stderr,"nnz nums overflow!!!:%ld\n",(uint64_t )ld * m);
        exit(EXIT_FAILURE);
    }
    ALPHA_Number *values = alpha_memalign((uint64_t)ld * m * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    ALPHA_INT *indices = alpha_memalign((uint64_t)ld * m * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    memset(values,0,(uint64_t)ld * m * sizeof(ALPHA_Number));
    memset(indices,0,(uint64_t)ld * m * sizeof(ALPHA_INT));

    const ALPHA_INT thread_num = alpha_get_thread_num();
    // i列j行, 列优先
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(thread_num)
    #endif
    for (ALPHA_INT i = 0; i < ld; i++)
    {
        for (ALPHA_INT j = 0; j < m; j++)
        {
            ALPHA_INT csr_rs = csr->rows_start[j];
            ALPHA_INT csr_re = csr->rows_end[j];
            if (csr_rs + i < csr_re)
            {
                values[i * m + j] = csr->values[csr_rs + i];
                indices[i * m + j] = csr->col_indx[csr_rs + i];
            }
        }
    }
    mat->values = values;
    mat->indices = indices;
// #ifndef COMPLEX
// #ifndef DOUBLE    
//     print_ell_s(mat);
// #endif
// #endif

    mat->d_values  = NULL;
    mat->d_indices = NULL;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
