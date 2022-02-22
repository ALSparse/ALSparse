#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>
#include "alphasparse/util/random.h"

static void mkl_add(const int argc, const char *argv[], const char *file, float alpha, sparse_index_base_t *ret_index, MKL_INT *ret_rows, MKL_INT *ret_cols, MKL_INT **ret_rows_start, MKL_INT **ret_rows_end, MKL_INT **ret_col_index, float **ret_values, int thread_num, MKL_INT row, MKL_INT col, float val)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    float *values;
    mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_s(values, 1, nnz);
    mkl_set_num_threads(thread_num);

    sparse_operation_t transA = mkl_args_get_transA(argc, argv);

    sparse_matrix_t coo, csr, csrt, result;
    mkl_call_exit(mkl_sparse_s_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "mkl_sparse_s_create_coo");
    mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr), "mkl_sparse_convert_csr");
    mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_TRANSPOSE, &csrt), "mkl_sparse_convert_csr");

    mkl_sparse_s_set_value(csr, row, col, val);

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_call_exit(mkl_sparse_s_add(transA, csr, alpha, csrt, &result), "mkl_sparse_s_add");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    mkl_sparse_order(result);

    mkl_call_exit(mkl_sparse_s_export_csr(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "mkl_sparse_s_export_csr");

    mkl_sparse_destroy(coo);
    mkl_sparse_destroy(csr);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_add(const int argc, const char *argv[], const char *file, float alpha, alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, float **ret_values, int thread_num, ALPHA_INT row, ALPHA_INT col, float val)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    float *values;
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_s(values, 1, nnz);
    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csr, csrt, result;

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr), "alphasparse_convert_csr");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_TRANSPOSE, &csrt), "alphasparse_convert_csr");

    alphasparse_s_set_value_plain(csr, row, col, val);

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_add_plain(transA, csr, alpha, csrt, &result), "alphasparse_s_add_plain");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    alpha_call_exit(alphasparse_s_export_csr(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_s_export_csr");

    alphasparse_destroy(coo);
    alphasparse_destroy(csr);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    const float alpha = 3.;

    // return
    sparse_index_base_t mkl_index;
    MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index, mkl_s_row, mkl_s_col;
    float *mkl_values;

    alphasparse_index_base_t alpha_index;
    ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index, alpha_s_row, alpha_s_col;
    float *alpha_values;

    srand(1);
    float random_values = random_float();
    int r = random_int(1000);
    int c = random_int(1000);
    mkl_s_row = r;
    mkl_s_col = c;
    alpha_s_row = r;
    alpha_s_col = c;

    alpha_add(argc, argv, file, alpha, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values, thread_num, alpha_s_row, alpha_s_col, random_values);

    int status = 0;
    if (check)
    {
        printf(",");
        mkl_add(argc, argv, file, alpha, &mkl_index, &mkl_rows, &mkl_cols, &mkl_rows_start, &mkl_rows_end, &mkl_col_index, &mkl_values, thread_num, mkl_s_row, mkl_s_col, random_values);
        int mkl_nnz = mkl_rows_end[mkl_rows - 1];
        int alpha_nnz = alpha_rows_end[alpha_rows - 1];
        printf(",");
        status = check_s(mkl_values, mkl_nnz, alpha_values, alpha_nnz);
    }
    printf("\n");
    return status;
}