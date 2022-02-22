/**
 * @brief openspblas spmm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static sparse_status_t alpha_convert_mkl_csc_z(alphasparse_matrix_t src, sparse_matrix_t *dst)
{
    spmat_csc_z_t * mat = (spmat_csc_z_t *)src->mat;
    sparse_status_t st =  mkl_sparse_z_create_csc(
        dst,
        SPARSE_INDEX_BASE_ZERO,
        mat->rows,
        mat->cols,
        mat->cols_start,
        mat->cols_end,
        mat->row_indx,
        (MKL_Complex16*) mat->values
    );
    return st;
}

static void mkl_spmm(const int argc, const char *argv[], const char *file, int thread_num, sparse_index_base_t *ret_index, MKL_INT *ret_rows, MKL_INT *ret_cols, MKL_INT **ret_rows_start, MKL_INT **ret_rows_end, MKL_INT **ret_col_index, MKL_Complex16 **ret_values)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex16 *values;
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_z(values, 1, nnz);

    mkl_set_num_threads(thread_num);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    
    alphasparse_matrix_t coo, alpha_csc;
    sparse_matrix_t csc,result;
    alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_csc), "alphasparse_convert_csc");
    alpha_convert_mkl_csc_z(alpha_csc, &csc);

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_call_exit(mkl_sparse_spmm(transA, csc, csc, &result), "mkl_sparse_spmm");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    mkl_sparse_order(result);

    mkl_call_exit(mkl_sparse_z_export_csc(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "mkl_sparse_z_export_csr");

    alphasparse_destroy(coo);
    alphasparse_destroy(alpha_csc);
    mkl_sparse_destroy(csc);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_spmm(const int argc, const char *argv[], const char *file, int thread_num, alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, ALPHA_Complex16 **ret_values)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex16 *values;
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_z(values, 1, nnz);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csr;
    alphasparse_matrix_t alpha_result;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_spmm_plain(transA, csr, csr, &alpha_result), "alphasparse_spmm_plain");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    alpha_call_exit(alphasparse_z_export_csc(alpha_result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_z_export_csr");

    alphasparse_destroy(coo);
    alphasparse_destroy(csr);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    // return
    sparse_index_base_t mkl_index;
    MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index;
    MKL_Complex16 *mkl_values;

    alphasparse_index_base_t alpha_index;
    ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
    ALPHA_Complex16 *alpha_values;

    alpha_spmm(argc, argv, file, thread_num, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values);

    int status = 0;
    if (check)
    {
        mkl_spmm(argc, argv, file, thread_num, &mkl_index, &mkl_rows, &mkl_cols, &mkl_rows_start, &mkl_rows_end, &mkl_col_index, &mkl_values);
        int mkl_nnz = mkl_rows_end[mkl_rows - 1];
        int alpha_nnz = alpha_rows_end[alpha_rows - 1];
        status = check_d((double *)mkl_values, 2 * mkl_nnz, (double *)alpha_values, 2 * alpha_nnz);
    }

    printf("\n");
    return status;
}