
/**
 * @brief openspblas mm bsr test
 * @author Pinjie Xu
 */

#include <alphasparse.h>
#include <stdio.h>

void alpha_mm(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex16 *values, ALPHA_Complex16 alpha, ALPHA_Complex16 beta, ALPHA_Complex16 *x, ALPHA_INT columns, ALPHA_INT ldx, ALPHA_Complex16 *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t coo, bsr;
    ALPHA_INT block_size = 4;
    alphasparse_layout_t block_layout = layout;
    alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(coo, block_size, block_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsr), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_mm(transA, alpha, bsr, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_z_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_z_mm");
    //alpha_timing_gflops_print(&timer, (ALPHA_Complex16)nnz * n * 2 + m * n, "alphasparse_z_mm");

    alphasparse_destroy(coo);
    alphasparse_destroy(bsr);
}

void alpha_mm_plain(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex16 *values, ALPHA_Complex16 alpha, ALPHA_Complex16 beta, ALPHA_Complex16 *x, ALPHA_INT columns, ALPHA_INT ldx, ALPHA_Complex16 *y, ALPHA_INT ldy, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t coo, bsr;
    ALPHA_INT block_size = 4;
    alphasparse_layout_t bl_layout = layout;
    alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(coo, block_size, bl_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsr), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_mm_plain(transA, alpha, bsr, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_z_mm_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_z_mm_plain");
    //alpha_timing_gflops_print(&timer, (ALPHA_Complex16)nnz * n * 2 + m * n, "alphasparse_z_mm_plain");

    alphasparse_destroy(coo);
    alphasparse_destroy(bsr);
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
    const ALPHA_Complex16 alpha = {.real = 3., .imag = 3.};
    const ALPHA_Complex16 beta = {.real = 3., .imag = 3.};

    // read coo
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

    ALPHA_INT columns = args_get_columns(argc, argv, k);

    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    ALPHA_INT ldx = columns, ldy = columns;
    ALPHA_INT rowsx = k, rowsy = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE || transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        rowsx = m;
        rowsy = k;
    }
    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = rowsx;
        ldy = rowsy;
    }
    ALPHA_INT sizex = rowsx * columns;
    ALPHA_INT sizey = rowsy * columns;

    ALPHA_Complex16* x = alpha_memalign(sizeof(ALPHA_Complex16) * sizex, DEFAULT_ALIGNMENT);
    ALPHA_Complex16* icty = alpha_memalign(sizeof(ALPHA_Complex16) * sizey, DEFAULT_ALIGNMENT);
    ALPHA_Complex16* icty_plain = alpha_memalign(sizeof(ALPHA_Complex16) * sizey, DEFAULT_ALIGNMENT);

    alpha_fill_random_z(x, 2, sizex);
    alpha_fill_random_z(icty, 1, sizey);

    printf("thread_num : %d\n", thread_num);

    alpha_mm(argc, argv, m, k, nnz, row_index, col_index, values, alpha, beta, x, columns, ldx, icty, ldy, thread_num);
    int status = 0;
    if (check)
    {
        alpha_fill_random_z(icty_plain, 1, sizey);
        alpha_mm_plain(argc, argv, m, k, nnz, row_index, col_index, values, alpha, beta, x, columns, ldx, icty_plain, ldy, thread_num);
        status = check_z(icty, sizey, icty_plain, sizey);
    }
    alpha_free(x);
    alpha_free(icty);
    alpha_free(icty_plain);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}
