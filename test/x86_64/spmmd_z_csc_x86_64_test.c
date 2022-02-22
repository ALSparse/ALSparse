/**
 * @brief openspblas spmmd csr test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static void mkl_spmmd(const int argc, const char *argv[], const char *file, int thread_num, MKL_Complex16 **ret, size_t *ret_size)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    MKL_Complex16 *values;
    mkl_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz * 2);
    if (m != k)
    {
        printf("m != k test is not support yet!!!");
        exit(-1);
    }

    size_t size_C = m * m;
    MKL_Complex16 *C = alpha_malloc(sizeof(MKL_Complex16) * size_C);
    MKL_INT ldc = m;

    mkl_set_num_threads(thread_num);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    sparse_matrix_t cooA, csrA;
    mkl_sparse_z_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    mkl_sparse_z_spmmd(transA, csrA, csrA, layout, C, ldc);
    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer,(double)nnz * k * 2 + m * k));
    mkl_sparse_destroy(cooA);
    mkl_sparse_destroy(csrA);

    *ret = C;
    *ret_size = size_C;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_spmmd(const int argc, const char *argv[], const char *file, int thread_num, ALPHA_Complex16 **ret, size_t *ret_size)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex16 *values;
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_z(values, 1, nnz);
    if (m != k)
    {
        printf("m != k test is not support yet!!!");
        exit(-1);
    }
    size_t size_C = m * m;
    ALPHA_Complex16 *C = alpha_malloc(sizeof(ALPHA_Complex16) * size_C);
    MKL_INT ldc = m;

    alpha_set_thread_num(thread_num);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alphasparse_matrix_t cooA, csrA;
    alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_z_spmmd_plain(transA, csrA, csrA, layout, C, ldc), "alphasparse_z_spmmd");
    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer,(double)nnz * k * 2 + m * k));

    *ret = C;
    *ret_size = size_C;
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

    ALPHA_Complex16 *alpha_C;
    MKL_Complex16 *mkl_C;
    size_t size_mkl_C, size_alpha_C;

    int status = 0;
    alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C);

    if (check)
    {
        mkl_spmmd(argc, argv, file, thread_num, &mkl_C, &size_mkl_C);
        status = check_d((double *)mkl_C, 2*size_mkl_C, (double *)alpha_C, 2*size_alpha_C);
        alpha_free(mkl_C);
    }

    alpha_free(alpha_C);
    printf("\n");
    return status;
}