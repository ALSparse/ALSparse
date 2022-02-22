/**
 * @brief openspblas mv csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>

static void alpha_mv(const int argc, const char *argv[],  ALPHA_INT m,  ALPHA_INT n,  ALPHA_INT nnz,  ALPHA_INT *row_index,  ALPHA_INT *col_index,  ALPHA_Complex8 *values,  ALPHA_Complex8 *x,  ALPHA_Complex8 alpha,  ALPHA_Complex8 beta, ALPHA_Complex8 *y, ALPHA_INT thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_c_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_c_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_c_mv(transA, alpha, cscA, descr, x, beta, y), "alphasparse_c_mv");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alpha_mv_c_csc_time");

    // save run time
    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }
    
    alphasparse_destroy(cooA);
    alphasparse_destroy(cscA);
}

static void alpha_mv_plain(const int argc, const char *argv[],  ALPHA_INT m,  ALPHA_INT n,  ALPHA_INT nnz,  ALPHA_INT *row_index,  ALPHA_INT *col_index,  ALPHA_Complex8 *values,  ALPHA_Complex8 *x,  ALPHA_Complex8 alpha,  ALPHA_Complex8 beta, ALPHA_Complex8 *y, ALPHA_INT thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_c_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_c_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_c_mv_plain(transA, alpha, cscA, descr, x, beta, y), "alphasparse_c_mv_plain");

    alpha_timing_end(&timer);

    alpha_timing_elaped_time_print(&timer, "alphasparse_c_mv_plain");

    alphasparse_destroy(cooA);
    alphasparse_destroy(cscA);
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
    ALPHA_Complex8 *values;
    const ALPHA_Complex8 alpha = {.real = 3., .imag = 3.};
    const ALPHA_Complex8 beta = {.real = 2., .imag = 2.};
    // read coo
    alpha_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);

    ALPHA_INT mm, kk; //是否转置
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE || transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
        mm = k;
        kk = m;
    }
    else{
        mm = m;
        kk = k;
    }
    
    // init x y
    ALPHA_Complex8 *x = alpha_malloc(kk * sizeof(ALPHA_Complex8));
    ALPHA_Complex8 *alpha_y = alpha_malloc(mm * sizeof(ALPHA_Complex8));
    ALPHA_Complex8 *alpha_y_plain = alpha_malloc(mm * sizeof(ALPHA_Complex8));

    alpha_fill_random_c(x, 0, kk);
    alpha_fill_random_c(alpha_y, 1, mm);

    printf("thread_num : %d\n", thread_num);

    alpha_mv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, beta, alpha_y, thread_num);

    int status = 0;
    if (check)
    {
        alpha_fill_random_c(alpha_y_plain, 1, mm);
        alpha_mv_plain(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, beta, alpha_y_plain, thread_num);
        status = check_c(alpha_y, mm, alpha_y_plain, mm);
    }

    alpha_free(x);
    alpha_free(alpha_y);
    alpha_free(alpha_y_plain);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}