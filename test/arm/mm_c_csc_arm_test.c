
/**
 * @brief openspblas mm csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>

void alpha_mm(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex8 *values, ALPHA_Complex8 alpha, ALPHA_Complex8 beta, ALPHA_Complex8 *x, ALPHA_INT columns, ALPHA_INT ldx, ALPHA_Complex8 *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t coo, csc;
    alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_c_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csc), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_c_mm(transA, alpha, csc, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_c_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_c_mm");
    alpha_timing_gflops_print(&timer, (double)nnz * n * 2 + m * n, "alphasparse_c_mm");

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

void alpha_mm_plain(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex8 *values, ALPHA_Complex8 alpha, ALPHA_Complex8 beta, ALPHA_Complex8 *x, ALPHA_INT columns, ALPHA_INT ldx, ALPHA_Complex8 *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t coo, csc;
    alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_c_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csc), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_c_mm_plain(transA, alpha, csc, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_c_mm_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_c_mm_plain");
    alpha_timing_gflops_print(&timer, (double)nnz * n * 2 + m * n, "alphasparse_c_mm_plain");

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
    ALPHA_Complex8 *values;
    const ALPHA_Complex8 alpha = {.real = 3., .imag = 3.};
    const ALPHA_Complex8 beta = {.real = 2., .imag = 2.};

    // read coo
    alpha_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);
    ALPHA_INT mm, kk;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE){
        mm = k;
        kk = m;
    }
    else{
        mm = m;
        kk = k;
    }

    ALPHA_INT columns = args_get_columns(argc, argv, k);

    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    ALPHA_INT ldx, ldy;
    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        ldx = columns;
        ldy = columns;
    }
    else
    {
        ldx = kk;
        ldy = mm;
    }

    ALPHA_Complex8 *x = alpha_memalign(sizeof(ALPHA_Complex8) * kk * columns, DEFAULT_ALIGNMENT);
    ALPHA_Complex8 *maly = alpha_memalign(sizeof(ALPHA_Complex8) * mm * columns, DEFAULT_ALIGNMENT);
    ALPHA_Complex8 *maly_plain = alpha_memalign(sizeof(ALPHA_Complex8) * mm * columns, DEFAULT_ALIGNMENT);

    alpha_fill_random_c(values, 2, nnz);
    alpha_fill_random_c(x, 2, kk * columns);
    alpha_fill_random_c(maly, 1, mm * columns);

    printf("thread_num : %d\n", thread_num);

    alpha_mm(argc, argv, m, k, nnz, row_index, col_index, values, alpha, beta, x, columns, ldx, maly, ldy, thread_num);
    int status = 0;
    if (check)
    {
        alpha_fill_random_c(maly_plain, 1, mm * columns);
        alpha_mm_plain(argc, argv, m, k, nnz, row_index, col_index, values, alpha, beta, x, columns, ldx, maly_plain, ldy, thread_num);
        status = check_c(maly, mm * columns, maly_plain, mm * columns);
    }
    alpha_free(x);
    alpha_free(maly);
    alpha_free(maly_plain);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}