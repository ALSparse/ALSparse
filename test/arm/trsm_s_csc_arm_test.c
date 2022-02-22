/**
 * @brief openspblas trsm csc test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>

void plain_trsm(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, float *values, float alpha, float *x, ALPHA_INT columns, ALPHA_INT ldx, float *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_trsm_plain(transA, alpha, cscA, descr, layout, x, columns, ldx, y, ldy), "alphasparse_s_trsm_plain");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer,(double)nnz * n * 2 + m * n));
    alphasparse_destroy(cooA);
    alphasparse_destroy(cscA);
}

void alpha_trsm(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, float *values, float alpha, float *x, ALPHA_INT columns, ALPHA_INT ldx, float *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_trsm(transA, alpha, cscA, descr, layout, x, columns, ldx, y, ldy), "alphasparse_s_trsm");

    alpha_timing_end(&timer);
    printf("%lf,%lf", alpha_timing_elapsed_time(&timer), alpha_timing_gflops(&timer,(double)nnz * n * 2 + m * n));
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
    float *values;
    float *x, *alpha_y, *plain_y;

    const float alpha = 2;
    // read coo
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

    int columns = args_get_columns(argc, argv, k);

    // init x y
    x = alpha_malloc(k * columns * sizeof(float));
    alpha_y = alpha_malloc(m * columns * sizeof(float));
    plain_y = alpha_malloc(m * columns * sizeof(float));

    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    int ldx, ldy;
    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        ldx = columns;
        ldy = columns;
    }
    else
    {
        ldx = k;
        ldy = m;
    }

    alpha_fill_random_s(x, 0, k * columns);

    printf("%d,", thread_num);

    int status = 0;
    alpha_trsm(argc, argv, m, k, nnz, row_index, col_index, values, alpha, x, columns, ldx, alpha_y, ldy, thread_num);

    if (check)
    {
        printf(",");
        plain_trsm(argc, argv, m, k, nnz, row_index, col_index, values, alpha, x, columns, ldx, plain_y, ldy, thread_num);
        printf(",");
        status = check_s(plain_y, m * columns, alpha_y, m * columns);
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
