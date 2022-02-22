/**
 * @brief openspblas trsm bsr test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>

void plain_trsm(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex16 *values, ALPHA_Complex16 alpha, ALPHA_Complex16 *x, ALPHA_INT columns, ALPHA_INT ldx, ALPHA_Complex16 *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, bsrA;
    const ALPHA_INT block_size = 2;
    const alphasparse_layout_t b_layout = layout;

    alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(cooA, block_size , b_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_trsm_plain(transA, alpha, bsrA, descr, layout, x, columns, ldx, y, ldy), "alphasparse_z_trsm_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_z_trsm_plain");
    alphasparse_destroy(cooA);
    alphasparse_destroy(bsrA);
}

void alpha_trsm(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex16 *values, ALPHA_Complex16 alpha, ALPHA_Complex16 *x, ALPHA_INT columns, ALPHA_INT ldx, ALPHA_Complex16 *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, bsrA;
    const ALPHA_INT block_size = 2;
    const alphasparse_layout_t b_layout = layout;
    alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(cooA, block_size , b_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_trsm(transA, alpha, bsrA, descr, layout, x, columns, ldx, y, ldy), "alphasparse_z_trsm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_z_trsm");
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
    ALPHA_Complex16 *values;
    ALPHA_Complex16 *x, *alpha_y, *plain_y;

    const ALPHA_Complex16 alpha = {.real=2 , .imag=2};
    // read coo
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

    int columns = args_get_columns(argc, argv, k);

    // init x y
    x = alpha_malloc(k * columns * sizeof(ALPHA_Complex16));
    alpha_y = alpha_malloc(m * columns * sizeof(ALPHA_Complex16));
    plain_y = alpha_malloc(m * columns * sizeof(ALPHA_Complex16));

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

    alpha_fill_random_z(x, 0, k * columns);

    printf("thread_num : %d\n", thread_num);

    int status = 0;
    alpha_trsm(argc, argv, m, k, nnz, row_index, col_index, values, alpha, x, columns, ldx, alpha_y, ldy, thread_num);

    if (check)
    {
        plain_trsm(argc, argv, m, k, nnz, row_index, col_index, values, alpha, x, columns, ldx, plain_y, ldy, thread_num);
        status = check_z(plain_y, m * columns, alpha_y, m * columns);
    }

    alpha_free(alpha_y);
    alpha_free(plain_y);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}
