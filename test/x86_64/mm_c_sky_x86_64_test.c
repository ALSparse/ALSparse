/**
 * @brief openspblas mm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static void mkl_mm(const int argc, const char *argv[], const char *file, int thread_num, MKL_Complex8 alpha, MKL_Complex8 beta, MKL_Complex8 **ret_y, size_t *ret_size_y)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    MKL_Complex8 *values;
    mkl_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);

    MKL_INT columns = args_get_columns(argc, argv, k);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    MKL_INT rowsx = k, rowsy = m;
    if (transA == SPARSE_OPERATION_TRANSPOSE || transA == SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        rowsx = m;
        rowsy = k;
    }
    MKL_INT ldx = columns, ldy = columns;
    if (layout == SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = rowsx;
        ldy = rowsy;
    }

    size_t size_x = rowsx * columns;
    size_t size_y = rowsy * columns;
    MKL_Complex8 *x = alpha_memalign(sizeof(MKL_Complex8) * size_x, DEFAULT_ALIGNMENT);
    MKL_Complex8 *y = alpha_memalign(sizeof(MKL_Complex8) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_c((ALPHA_Complex8 *)x, 1, size_x);
    alpha_fill_random_c((ALPHA_Complex8 *)y, 1, size_y);

    mkl_set_num_threads(thread_num);
    sparse_matrix_t coo, csr;
    mkl_call_exit(mkl_sparse_c_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "mkl_sparse_c_create_coo");
    mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr), "mkl_sparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_call_exit(mkl_sparse_c_mm(transA, alpha, csr, descr, layout, x, columns, ldx, beta, y, ldy), "mkl_sparse_c_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_c_mm");

    *ret_y = y;
    *ret_size_y = size_y;

    mkl_sparse_destroy(coo);
    mkl_sparse_destroy(csr);
    alpha_free(x);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_mm(const int argc, const char *argv[], const char *file, int thread_num, ALPHA_Complex8 alpha, ALPHA_Complex8 beta, ALPHA_Complex8 **ret_y, size_t *ret_size_y)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex8 *values;
    alpha_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);

    ALPHA_INT columns = args_get_columns(argc, argv, k);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    ALPHA_INT rowsx = k, rowsy = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE || transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        rowsx = m;
        rowsy = k;
    }
    ALPHA_INT ldx = columns, ldy = columns;
    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = rowsx;
        ldy = rowsy;
    }
    size_t size_x = rowsx * columns;
    size_t size_y = rowsy * columns;
    ALPHA_Complex8 *x = alpha_memalign(sizeof(ALPHA_Complex8) * size_x, DEFAULT_ALIGNMENT);
    ALPHA_Complex8 *y = alpha_memalign(sizeof(ALPHA_Complex8) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_c(x, 1, size_x);
    alpha_fill_random_c(y, 1, size_y);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csr;
    alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_c_create_coo");
    alpha_call_exit(alphasparse_convert_sky(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, descr.mode, &csr), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_c_mm_plain(transA, alpha, csr, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_c_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_c_mm");

    alphasparse_destroy(coo);
    alphasparse_destroy(csr);

    *ret_y = y;
    *ret_size_y = size_y;

    alpha_free(x);
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

    const ALPHA_Complex8 alpha = {.real = 3., .imag = 3.};
    const ALPHA_Complex8 beta = {.real = 2., .imag = 2.};

    const MKL_Complex8 mkl_alpha = {.real = 3., .imag = 3.};
    const MKL_Complex8 mkl_beta = {.real = 2., .imag = 2.};

    printf("thread_num : %d\n", thread_num);

    ALPHA_Complex8 *alpha_y;
    MKL_Complex8 *mkl_y;
    size_t size_alpha_y, size_mkl_y;

    alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);

    int status = 0;
    if (check)
    {
        mkl_mm(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y, &size_mkl_y);
        check_s((float *)mkl_y, size_mkl_y * 2, (float *)alpha_y, size_alpha_y * 2);
        alpha_free(mkl_y);
    }

    alpha_free(alpha_y);
    return status;
}