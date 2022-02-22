
/**
 * @brief openspblas mm csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>

void alpha_mm(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, double *values, double alpha, double beta, double *x, ALPHA_INT columns, ALPHA_INT ldx, double *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t coo, csc;
    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csc), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    printf("alphasparse_d_mm start !!! \n");
    alpha_call_exit(alphasparse_d_mm(transA, alpha, csc, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_d_mm");
    printf("alphasparse_d_mm end !!! \n");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_mm");
    alpha_timing_gflops_print(&timer, (double)nnz * n * 2 + m * n, "alphasparse_d_mm");

    // save run time
    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }

    alphasparse_destroy(coo);
    alphasparse_destroy(csc);
}

void alpha_mm_plain(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, double *values, double alpha, double beta, double *x, ALPHA_INT columns, ALPHA_INT ldx, double *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t coo, csc;
    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csc), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_d_mm_plain(transA, alpha, csc, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_d_mm_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_mm_plain");
    alpha_timing_gflops_print(&timer, (double)nnz * n * 2 + m * n, "alphasparse_d_mm_plain");

    alphasparse_destroy(coo);
    alphasparse_destroy(csc);
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
    double *values;
    const double alpha = 3.;
    const double beta = 2.;

    // read coo
    alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
    
    ALPHA_INT columns = args_get_columns(argc, argv, k);

    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    ALPHA_INT x_rows, y_rows;
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
        x_rows = k;
        y_rows = m;
    }
    else{ // 转置&共轭转置
        x_rows = m;
        y_rows = k;
    }

    ALPHA_INT ldx, ldy;
    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        ldx = columns;
        ldy = columns;
    }
    else
    {
        ldx = x_rows;
        ldy = y_rows;
    }

    double *x = alpha_memalign(sizeof(double) * x_rows * columns, DEFAULT_ALIGNMENT);
    double *openspblasy = alpha_memalign(sizeof(double) * y_rows * columns, DEFAULT_ALIGNMENT);
    double *openspblasy_plain = alpha_memalign(sizeof(double) * y_rows * columns, DEFAULT_ALIGNMENT);

    alpha_fill_random_d(values, 2, nnz);
    alpha_fill_random_d(x, 2, x_rows * columns);
    alpha_fill_random_d(openspblasy, 1, y_rows * columns);

    printf("thread_num : %d\n", thread_num);

    alpha_mm(argc, argv, m, k, nnz, row_index, col_index, values, alpha, beta, x, columns, ldx, openspblasy, ldy, thread_num);
    int status = 0;
    if (check)
    {
        alpha_fill_random_d(openspblasy_plain, 1, y_rows * columns);
        alpha_mm_plain(argc, argv, m, k, nnz, row_index, col_index, values, alpha, beta, x, columns, ldx, openspblasy_plain, ldy, thread_num);
        status = check_d(openspblasy, y_rows * columns, openspblasy_plain, y_rows * columns);
    }
    alpha_free(x);
    alpha_free(openspblasy);
    alpha_free(openspblasy_plain);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}
