/**
 * @brief openspblas trsv coo test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>

void plain_trsv(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, float *values, float *x, float alpha, float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_s_create_coo");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_trsv_plain(transA, alpha, cooA, descr, x, y), "alphasparse_s_trsv_plain");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer,(double)nnz * n * 2 + m * n));
    alphasparse_destroy(cooA);
}

void alpha_trsv(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, float *values, float *x, float alpha, float *y, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_s_create_coo");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_trsv(transA, alpha, cooA, descr, x, y), "alphasparse_s_trsv");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer,(double)nnz * n * 2 + m * n));
    alphasparse_destroy(cooA);
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    float *values;
    float *x, *alpha_y, *plain_y;
    const float alpha = 2.;

    // read coo
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

    // init x y
    x = alpha_malloc(k * sizeof(float));
    alpha_y = alpha_malloc(m * sizeof(float));
    plain_y = alpha_malloc(m * sizeof(float));

    // alpha_fill_random_s(x, 0, k);
    alpha_fill_random_s(x, 1, k);

    printf("%d,\n", thread_num);

    alpha_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, alpha_y, thread_num);

    int status = 0;
    if (check)
    {
        plain_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, plain_y, thread_num);
        status = check_s(plain_y, m, alpha_y, m);
    }

    alpha_free(x);
    alpha_free(alpha_y);
    alpha_free(plain_y);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    printf("\n");
    return status;
}