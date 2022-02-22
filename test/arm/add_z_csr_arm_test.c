#include <alphasparse.h>
#include <stdio.h>

static void alpha_add_plain(const int argc, const char *argv[], const char *file, ALPHA_Complex16 alpha, alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, ALPHA_Complex16 **ret_values, int thread_num)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex16 *values;
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_z(values, 1, nnz);
    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csr, csrt, result;

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr), "alphasparse_convert_csr");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_TRANSPOSE, &csrt), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_add_plain(transA, csr, alpha, csrt, &result), "alphasparse_z_add_plain");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    alpha_call_exit(alphasparse_z_export_csr(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_z_export_csr");

    alphasparse_destroy(coo);
    alphasparse_destroy(csr);
    alphasparse_destroy(csrt);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_add(const int argc, const char *argv[], const char *file, ALPHA_Complex16 alpha, alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, ALPHA_Complex16 **ret_values, int thread_num)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex16 *values;
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_z(values, 1, nnz);
    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csr, csrt, result;

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr), "alphasparse_convert_csr");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_TRANSPOSE, &csrt), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_add_plain(transA, csr, alpha, csrt, &result), "alphasparse_z_add_plain");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    alpha_call_exit(alphasparse_z_export_csr(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_z_export_csr");

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

    const ALPHA_Complex16 alpha = {.real = 3., .imag = 3.};
    const ALPHA_Complex16 beta = {.real = 3., .imag = 3.};

    // return
    alphasparse_index_base_t plain_index;
    ALPHA_INT plain_rows, plain_cols, *plain_rows_start, *plain_rows_end, *plain_col_index;
    ALPHA_Complex16 *plain_values;

    alphasparse_index_base_t alpha_index;
    ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
    ALPHA_Complex16 *alpha_values;
    printf("%d,", thread_num);

    alpha_add(argc, argv, file, alpha, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values, thread_num);

    int status = 0;
    if (check)
    {
        printf(",");
        alpha_add_plain(argc, argv, file, alpha, &plain_index, &plain_rows, &plain_cols, &plain_rows_start, &plain_rows_end, &plain_col_index, &plain_values, thread_num);
        int plain_nnz = plain_rows_end[plain_rows - 1];
        int alpha_nnz = alpha_rows_end[alpha_rows - 1];
        printf(",");
        // status = check_z(plain_values, plain_nnz, alpha_values, alpha_nnz);
        status = check_z_l3((ALPHA_Complex16 *)alpha_values, 0, alpha_nnz, alpha_values, 0, plain_nnz, alpha_col_index, NULL, 0, alpha_values, 0, alpha, beta, argc, argv);
    }
    printf("\n");

    return status;
}