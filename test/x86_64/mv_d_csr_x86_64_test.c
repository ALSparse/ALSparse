/**
 * @brief openspblas mv csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static void mkl_mv(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, const double beta, double **ret_y, size_t *ret_size_y)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    double *values;
    mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    size_t size_x = k;
    size_t size_y = m;
    if (transA == SPARSE_OPERATION_TRANSPOSE || transA == SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        size_x = m;
        size_y = k;
    }

    double *x = alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
    double *y = alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_d((double *)x, 1, size_x);
    alpha_fill_random_d((double *)y, 1, size_y);

    mkl_set_num_threads(thread_num);

    sparse_matrix_t cooA, csrA;
    mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_sparse_d_mv(transA, alpha, csrA, descr, x, beta, y);

    alpha_timing_end(&timer);

    printf("mkl time elapsed %lf s,gflops %lf\n", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

    mkl_sparse_destroy(cooA);
    mkl_sparse_destroy(csrA);

 *ret_y = y;
 *ret_size_y = size_y;

    alpha_free(x);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}
static void alpha_mv(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, const double beta, double **ret_y, size_t *ret_size_y, double **ret_x, size_t *ret_size_x)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    double *values;
    alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
    size_t size_x = k;
    size_t size_y = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE || transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        size_x = m;
        size_y = k;
    }
    double *x = alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
    double *y = alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_d(x, 1, size_x);
    alpha_fill_random_d(y, 1, size_y);

    alpha_set_thread_num(thread_num);

    alphasparse_matrix_t cooA, csrA;
    alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csr(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "alphasparse_convert_csr");

    alpha_clear_cache();

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_d_mv_plain(transA, alpha, csrA, descr, x, beta, y), "alphasparse_d_mv");

    alpha_timing_end(&timer);
    printf("openspblas time elapsed %lf s,gflops %lf\n", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));
    alphasparse_destroy(cooA);
    alphasparse_destroy(csrA);

 *ret_y = y;
 *ret_size_y = size_y;

 *ret_x = x;
 *ret_size_x = size_x;

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

    const double alpha = 2.f;
    const double beta = 2.f;

    const double mkl_alpha = 2.f;
    const double mkl_beta = 2.f;

    double *alpha_y;
    double *mkl_y;
    size_t size_alpha_y, size_mkl_y;
    double *alpha_x;
    size_t size_alpha_x;

    printf("thread=%d\n", thread_num);

    alpha_mv(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y, &alpha_x, &size_alpha_x);

    int status = 0;

    if (check)
    {
        // printf(",");
        mkl_mv(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y, &size_mkl_y);
        // printf(",");
//        status = check_d(double *)mkl_y, size_mkl_y * 2, (double *)alpha_y, size_alpha_y * 2);
        status = check_d_l2((double*)mkl_y, size_mkl_y, alpha_y, size_alpha_y, alpha_x, NULL, alpha, beta, argc, argv);
        alpha_free(mkl_y);
    }
    printf("\n");
    alpha_free(alpha_y);
    return status;
}