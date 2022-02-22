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

alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_HYB **dest)
{
    ALPHA_SPMAT_HYB *mat  = alpha_malloc(sizeof(ALPHA_SPMAT_HYB));
                  *dest = mat;

    ALPHA_SPMAT_CSR *csr;
    convert_csr_coo(source, &csr);

    ALPHA_INT m         = csr->rows;
    ALPHA_INT n         = csr->cols;
    ALPHA_INT csr_nnz   = source->nnz;
    ALPHA_INT ell_width = (csr_nnz - 1) / m + 1;

    ALPHA_INT coo_nnz = 0;
    for (ALPHA_INT i = 0; i < m; i++)
    {
        ALPHA_INT row_nnz  = csr->rows_end[i] - csr->rows_start[i];
        ALPHA_INT deta     = row_nnz - ell_width;
                coo_nnz += deta > 0 ? deta : 0;
    }

    const      ALPHA_INT thread_num = alpha_get_thread_num();
    ALPHA_Number *ell_values        = alpha_memalign((uint64_t)ell_width * m * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    ALPHA_INT    *ell_col_ind       = alpha_memalign((uint64_t)ell_width * m * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    ALPHA_Number *coo_values        = alpha_memalign((uint64_t)coo_nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    ALPHA_INT    *coo_row_val       = alpha_memalign((uint64_t)coo_nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    ALPHA_INT    *coo_col_val       = alpha_memalign((uint64_t)coo_nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);

    memset(ell_values, 0, (uint64_t)ell_width * m * sizeof(ALPHA_Number));
    memset(ell_col_ind, 0, (uint64_t)ell_width * m * sizeof(ALPHA_INT));
    memset(coo_values, 0, (uint64_t)coo_nnz * sizeof(ALPHA_Number));
    memset(coo_row_val, 0, (uint64_t)coo_nnz * sizeof(ALPHA_INT));
    memset(coo_col_val, 0, (uint64_t)coo_nnz * sizeof(ALPHA_INT));

    // i列j行, 列优先
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(thread_num)
    #endif
    for (ALPHA_INT i = 0; i < ell_width; i++)
    {
        for (ALPHA_INT j = 0; j < m; j++)
        {
            ALPHA_INT csr_rs = csr->rows_start[j];
            ALPHA_INT csr_re = csr->rows_end[j];
            if (csr_rs + i < csr_re)
            {
                ell_values  [i * m + j] = csr->values[csr_rs + i];
                ell_col_ind[i * m + j]  = csr->col_indx[csr_rs + i];
            }
        }
    }
    ALPHA_INT idx = 0;
    for (ALPHA_INT i = ell_width; i < n; i++)
    {
        for (ALPHA_INT j = 0; j < m; j++)
        {
            ALPHA_INT csr_rs = csr->rows_start[j];
            ALPHA_INT csr_re = csr->rows_end[j];
            if (csr_rs + i < csr_re)
            {
                coo_values  [idx] = csr->values[csr_rs + i];
                coo_row_val[idx]  = j;
                coo_col_val[idx]  = csr->col_indx[csr_rs +i];
                idx++;
            }
        }
    }

    // coo part sort
    ALPHA_SPMAT_COO *coo = alpha_malloc(sizeof(ALPHA_SPMAT_COO));
    coo->col_indx = coo_col_val;
    coo->row_indx = coo_row_val;
    coo->values = coo_values;
    coo->cols = n;
    coo->rows = m;
    coo->nnz = coo_nnz;
    // coo->ordered = false;
    // coo_order(coo);

    mat->ell_val     = ell_values;
    mat->ell_col_ind = ell_col_ind;
    mat->coo_val     = coo->values;
    mat->coo_col_val = coo->col_indx;
    mat->coo_row_val = coo->row_indx;
    mat->nnz         = coo_nnz;
    mat->rows        = m;
    mat->cols        = n;
    mat->ell_width   = ell_width;

//#define CHECK

#ifdef CHECK
#ifdef S
    print_coo_s(source);

    print_coo_s(coo);

    ALPHA_SPMAT_ELL *ell = alpha_malloc(sizeof(ALPHA_SPMAT_ELL));
    ell->values  = ell_values;
    ell->indices = ell_col_ind;
    ell->ld      = ell_width;
    ell->rows    = m;
    ell->cols    = n;
    print_ell_s(ell);
#endif
#undef CHECK
#endif
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
