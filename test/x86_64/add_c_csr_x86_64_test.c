#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static void mkl_add(const int argc, const char *argv[], const char *file, MKL_Complex8 alpha, sparse_index_base_t *ret_index, MKL_INT *ret_rows, MKL_INT *ret_cols, MKL_INT **ret_rows_start, MKL_INT **ret_rows_end, MKL_INT **ret_col_index, MKL_Complex8 **ret_values, int thread_num)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    MKL_Complex8 *values;
    mkl_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_c((ALPHA_Complex8*)values, 1, nnz);
    mkl_set_num_threads(thread_num);

    sparse_operation_t transA = mkl_args_get_transA(argc, argv);

    sparse_matrix_t coo, csr, csrt, result;
    mkl_call_exit(mkl_sparse_c_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "mkl_sparse_c_create_coo");
    mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr), "mkl_sparse_convert_csr");
    mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_TRANSPOSE, &csrt), "mkl_sparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_call_exit(mkl_sparse_c_add(transA, csr, alpha, csrt, &result), "mkl_sparse_c_add");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    mkl_sparse_order(result);

    mkl_call_exit(mkl_sparse_c_export_csr(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "mkl_sparse_c_export_csr");

    mkl_sparse_destroy(coo);
    mkl_sparse_destroy(csr);
    mkl_sparse_destroy(csrt);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_add(const int argc, const char *argv[], const char *file, ALPHA_Complex8 alpha, alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, ALPHA_Complex8 **ret_values, int thread_num)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex8 *values;
    alpha_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_c(values, 1, nnz);
    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csr, csrt, result;

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_c_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr), "alphasparse_convert_csr");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_TRANSPOSE, &csrt), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_c_add_plain(transA, csr, alpha, csrt, &result), "alphasparse_c_add_plain");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    alpha_call_exit(alphasparse_c_export_csr(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_c_export_csr");

    alphasparse_destroy(coo);
    alphasparse_destroy(csr);
    alphasparse_destroy(csrt);

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

    const ALPHA_Complex8 alpha_alpha = {.real = 3., .imag = 3.};
    const MKL_Complex8 mkl_alpha = {.real = 3., .imag = 3.};

    // return
    sparse_index_base_t mkl_index;
    MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index;
    MKL_Complex8 *mkl_values;

    alphasparse_index_base_t alpha_index;
    ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
    ALPHA_Complex8 *alpha_values;
    ALPHA_Complex8 alpha = {0, 0};
    ALPHA_Complex8 beta = {0, 0};
    alpha_add(argc, argv, file, alpha_alpha, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values, thread_num);

    int status = 0;
    if (check)
    {
        printf(",");
        mkl_add(argc, argv, file, mkl_alpha, &mkl_index, &mkl_rows, &mkl_cols, &mkl_rows_start, &mkl_rows_end, &mkl_col_index, &mkl_values, thread_num);
        int mkl_nnz = mkl_rows_end[mkl_rows - 1];
        int alpha_nnz = alpha_rows_end[alpha_rows - 1];
        printf(",");
//        status = check_s(float *)mkl_values, 2 * mkl_nnz, (float *)alpha_values, 2 * alpha_nnz);
        status = check_c_l3((ALPHA_Complex8 *)mkl_values, 0, mkl_nnz, alpha_values, 0, alpha_nnz, alpha_col_index, NULL, 0, alpha_values, 0, alpha, beta, argc, argv);
    }
    printf("\n");
    return status;
}