/**
 * @brief openspblas spmmd coo test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static void mkl_spmmd(const int argc, const char *argv[], const char *file, int thread_num, float **ret, size_t *ret_size)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    float *values;
    mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_s(values, 1, nnz);
    

    size_t size_C = m * m;
    float *C = alpha_malloc(sizeof(float) * size_C);
    MKL_INT ldc = m;

    mkl_set_num_threads(thread_num);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    sparse_matrix_t cooA;
    mkl_sparse_s_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    mkl_sparse_s_spmmd(transA, cooA, cooA, layout, C, ldc);
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_s_spmmd");
    mkl_sparse_destroy(cooA);

    *ret = C;
    *ret_size = size_C;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_spmmd(const int argc, const char *argv[], const char *file, int thread_num, float **ret, size_t *ret_size)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    float *values;
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_s(values, 1, nnz);
    
    size_t size_C = m * m;
    float *C = alpha_malloc(sizeof(float) * size_C);
    MKL_INT ldc = m;

    alpha_set_thread_num(thread_num);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alphasparse_matrix_t cooA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_s_spmmd_plain(transA, cooA, cooA, layout, C, ldc), "alphasparse_s_spmmd");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_spmmd");

    *ret = C;
    *ret_size = size_C;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    alphasparse_destroy(cooA);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    float *mkl_C, *alpha_C;
    size_t size_mkl_C, size_alpha_C;

    int status = 0;
    alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C);

    if (check)
    {
        mkl_spmmd(argc, argv, file, thread_num, &mkl_C, &size_mkl_C);
        status = check_s(mkl_C, size_mkl_C, alpha_C, size_alpha_C);
        alpha_free(mkl_C);
    }

    alpha_free(alpha_C);
    return status;
}