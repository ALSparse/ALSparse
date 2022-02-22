/**
 * @brief openspblas mv sky test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>

static void alpha_mv(const int argc, const char *argv[],  ALPHA_INT m,  ALPHA_INT n,  ALPHA_INT nnz,  ALPHA_INT *row_index,  ALPHA_INT *col_index,  float *values,  float *x,  float alpha,  float beta, float *y, ALPHA_INT thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, skyA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_sky(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, descr.mode, &skyA), "alphasparse_convert_sky");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_mv(transA, 2, skyA, descr, x, 3, y), "alphasparse_s_mv");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "openspblas mv s sky time");
    alphasparse_destroy(cooA);
    alphasparse_destroy(skyA);
}

static void alpha_mv_plain(const int argc, const char *argv[],  ALPHA_INT m,  ALPHA_INT n,  ALPHA_INT nnz,  ALPHA_INT *row_index,  ALPHA_INT *col_index,  float *values,  float *x,  float alpha,  float beta, float *y, ALPHA_INT thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, skyA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_sky(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, descr.mode, &skyA), "alphasparse_convert_sky");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_mv_plain(transA, 2, skyA, descr, x, 3, y), "alphasparse_s_mv_plain");

    alpha_timing_end(&timer);

    alpha_timing_elaped_time_print(&timer, "alphasparse_s_mv_plain");

    alphasparse_destroy(cooA);
    alphasparse_destroy(skyA);
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
    const float alpha = 2;
    const float beta = 3;
    // read coo
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    ALPHA_INT sizex = k,sizey = m;
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE || transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
        sizex = m;
        sizey = k;
    }
    // init x y
    float *x = alpha_malloc(sizex * sizeof(float));
    float *alpha_y = alpha_malloc(sizey * sizeof(float));
    float *alpha_y_plain = alpha_malloc(sizey * sizeof(float));

    alpha_fill_random_s(x, 0, sizex);
    alpha_fill_random_s(alpha_y, 1, sizey);

    printf("thread_num : %d\n", thread_num);

    alpha_mv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, beta, alpha_y, thread_num);

    int status = 0;
    if (check)
    {
        alpha_fill_random_s(alpha_y_plain, 1, sizey);
        alpha_mv_plain(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, beta, alpha_y_plain, thread_num);
        status = check_s(alpha_y, sizey, alpha_y_plain, sizey);
    }

    alpha_free(x);
    alpha_free(alpha_y);
    alpha_free(alpha_y_plain);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}