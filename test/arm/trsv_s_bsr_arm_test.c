/**
 * @brief openspblas trsv bsr test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>

#define ROW_MAJOR 0
#define COL_MAJOR 1

void plain_trsv(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, float *values, float *x, float alpha, float *y, int thread_num, int flag)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, bsrA;
    ALPHA_INT block_size = 4;
    alphasparse_layout_t bl_layout;
    if(flag == ROW_MAJOR)
        bl_layout = ALPHA_SPARSE_LAYOUT_ROW_MAJOR;
    else
        bl_layout = ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR;
    
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(cooA, block_size, bl_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_trsv_plain(transA, alpha, bsrA, descr, x, y), "alphasparse_s_trsv_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_trsv_plain");
    alphasparse_destroy(cooA);
    alphasparse_destroy(bsrA);
}

void alpha_trsv(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, float *values, float *x, float alpha, float *y, int thread_num, int flag)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, bsrA;
    ALPHA_INT block_size = 4;
    alphasparse_layout_t bl_layout;
    if(flag == ROW_MAJOR) bl_layout = ALPHA_SPARSE_LAYOUT_ROW_MAJOR;
    else bl_layout = ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(cooA, block_size, bl_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_trsv(transA, alpha, bsrA, descr, x, y), "alphasparse_s_trsv");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_trsv");
    alphasparse_destroy(cooA);
    alphasparse_destroy(bsrA);
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

    printf("thread_num : %d\n", thread_num);

    printf("ROW MAJOR\n");
    alpha_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, alpha_y, thread_num, ROW_MAJOR);

    int status = 0;
    if (check)
    {
        plain_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, plain_y, thread_num, ROW_MAJOR);
        status = check_s(plain_y, m, alpha_y, m);
    }
    printf("COL MAJOR\n");
    alpha_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, alpha_y, thread_num, COL_MAJOR);
    if (check)
    {
        plain_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, plain_y, thread_num, COL_MAJOR);
        status = check_s(plain_y, m, alpha_y, m);
    }

    alpha_free(alpha_y);
    alpha_free(plain_y);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}