/**
 * @brief openspblas spmmd csr test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>

static void alpha_spmmd_plain(const int argc, const char *argv[], const char *file, int thread_num, float **ret, size_t *ret_size)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    float *values;
    const char *fileA = args_get_data_fileA(argc, argv);
    alpha_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

    size_t size_C = m * m;
    float *C = alpha_malloc(sizeof(float) * size_C);
    ALPHA_INT ldc = m;

    alpha_set_thread_num(thread_num);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alphasparse_matrix_t coo, csrA, csrB, result;
    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "alphasparse_convert_csr");

    alphasparse_destroy(coo);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    const char *fileB;
    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo(fileB, &m, &k, &nnz, &row_index, &col_index, &values);

    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrB), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alphasparse_s_spmmd_plain(transA, csrA, csrB, layout, C, ldc);
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_spmmd_plain");
    alphasparse_destroy(coo);
    alphasparse_destroy(csrA);
    alphasparse_destroy(csrB);

 *ret = C;
 *ret_size = size_C;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_spmmd(const int argc, const char *argv[], const char *file, int thread_num, float **ret, size_t *ret_size,ALPHA_INT *ret_ldc)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    float *values;
    const char *fileA = args_get_data_fileA(argc, argv);
    alpha_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

    size_t size_C = m * m;
    float *C = alpha_malloc(sizeof(float) * size_C);
    ALPHA_INT ldc = m;

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csrA, csrB, result;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "alphasparse_convert_csr");
    alphasparse_destroy(coo);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    const char *fileB;
    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrB), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_s_spmmd(transA, csrA, csrB, layout, C, ldc), "alphasparse_s_spmmd");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_spmmd");

 *ret = C;
 *ret_size = size_C;
 *ret_ldc = ldc;
    alphasparse_destroy(coo);
    alphasparse_destroy(csrA);
    alphasparse_destroy(csrB);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = NULL; //args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    float *alpha_C, *alpha_S_plain;
    size_t size_alpha_C, size_alpha_S_plain;

    int status = 0;
    ALPHA_INT ldc;
    alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C,&ldc);
    float alpha = 0.f;
    float beta = 0.f;

    if (check)
    {
        alpha_spmmd_plain(argc, argv, file, thread_num, &alpha_S_plain, &size_alpha_S_plain);
        status = check_s_l3((float *)alpha_C, ldc, size_alpha_C, alpha_S_plain, ldc, size_alpha_S_plain, NULL, NULL, 0, alpha_C, ldc, alpha, beta, argc, argv);
        // status = check_s(alpha_C, size_alpha_C, alpha_S_plain, size_alpha_S_plain);
        alpha_free(alpha_S_plain);
    }

    alpha_free(alpha_C);
    return status;
}