/**
 * @brief openspblas spmm csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>

static void alpha_plain_spmm(const int argc, const char *argv[], alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, double **ret_values, int thread_num)
{
    ALPHA_INT m, n, nnz;
    ALPHA_INT *row_index, *col_index;
    double *values;
    const char *fileA = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileA, &m, &n, &nnz, &row_index, &col_index, &values);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, cscA, cscB, result;

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");

    alphasparse_destroy(coo);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    const char *fileB = NULL;
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileB, &m, &n, &nnz, &row_index, &col_index, &values);

    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_spmm_plain(transA, cscA, cscB, &result), "alphasparse_d_spmm_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_spmm_plain");

    alpha_call_exit(alphasparse_d_export_csc(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_d_export_csc");

    alphasparse_destroy(coo);
    alphasparse_destroy(cscA);
    alphasparse_destroy(cscB);
}

static void alpha_spmm(const int argc, const char *argv[], alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, double **ret_values, int thread_num)
{
    ALPHA_INT m, n, nnz;
    ALPHA_INT *row_index, *col_index;
    double *values;
    const char *fileA = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileA, &m, &n, &nnz, &row_index, &col_index, &values);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, cscA, cscB, result;

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");
    alphasparse_destroy(coo);   
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    const char *fileB = NULL;
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileB, &m, &n, &nnz, &row_index, &col_index, &values);
    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_spmm(transA, cscA, cscB, &result), "alphasparse_d_spmm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_spmm");

    alpha_call_exit(alphasparse_d_export_csc(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_d_export_csc");

    alphasparse_destroy(coo);
    alphasparse_destroy(cscA);
    alphasparse_destroy(cscB);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    // return
    alphasparse_index_base_t alpha_plain_index;
    ALPHA_INT alpha_plain_rows, alpha_plain_cols, *alpha_plain_rows_start, *alpha_plain_rows_end, *alpha_plain_col_index;
    double *alpha_plain_values;

    alphasparse_index_base_t alpha_index;
    ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
    double *alpha_values;

    alpha_spmm(argc, argv, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values, thread_num);

    int status = 0;
    if (check)
    {
        alpha_plain_spmm(argc, argv, &alpha_plain_index, &alpha_plain_rows, &alpha_plain_cols, &alpha_plain_rows_start, &alpha_plain_rows_end, &alpha_plain_col_index, &alpha_plain_values, thread_num);
        int alpha_plain_nnz = alpha_plain_rows_end[alpha_plain_rows - 1];
        int alpha_nnz = alpha_rows_end[alpha_rows - 1];

        status = check_d(alpha_plain_values, alpha_plain_nnz, alpha_values, alpha_nnz);
    }
    return status;
}