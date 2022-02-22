/**
 * @brief openspblas mm coo test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static void mkl_mm(const int argc, const char *argv[], const char *file, int thread_num, float alpha, float beta, float **ret_y, size_t *ret_size_y)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    float *values;
    sparse_index_base_t base = SPARSE_INDEX_BASE_ZERO;
    mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

    MKL_INT columns = args_get_columns(argc, argv, k);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    MKL_INT ldx = columns, ldy = columns;
    MKL_INT rows = m, cols = k;
    if(transA == SPARSE_OPERATION_TRANSPOSE){
        rows = k;
        cols = m;
    }

    if (layout == SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = cols;
        ldy = rows;
        base = SPARSE_INDEX_BASE_ONE;
    }

    if (base == SPARSE_INDEX_BASE_ONE)
    {
        for (int i=0; i<nnz; i++) 
        {
            row_index[i]++;
            col_index[i]++;
        }
    }
    size_t size_x = cols * columns;
    size_t size_y = rows * columns;
    float *x = alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
    float *y = alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_s((float *)values, 1,nnz);
    alpha_fill_random_s((float *)x, 1,size_x);
    alpha_fill_random_s((float *)y, 1,size_y);

    mkl_set_num_threads(thread_num);
    sparse_matrix_t coo;
    mkl_call_exit(mkl_sparse_s_create_coo(&coo, base, m, k, nnz, row_index, col_index, values), "mkl_sparse_s_create_coo");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_call_exit(mkl_sparse_s_mm(transA, alpha, coo, descr, layout, x, columns, ldx, beta, y, ldy), "mkl_sparse_s_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_s_mm");

    *ret_y = y;
    *ret_size_y = size_y;

    mkl_sparse_destroy(coo);
    alpha_free(x);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_mm(const int argc, const char *argv[], const char *file, int thread_num, float alpha, float beta, float **ret_y, size_t *ret_size_y)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    float *values;
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

    ALPHA_INT columns = args_get_columns(argc, argv, k);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    ALPHA_INT ldx = columns, ldy = columns;
    ALPHA_INT rows = m, cols = k;
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE){
        rows = k;
        cols = m;
    }

    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = cols;
        ldy = rows;
    }

    size_t size_x = cols * columns;
    size_t size_y = rows * columns;
    float *x = alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
    float *y = alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(values, 1, nnz);
    alpha_fill_random_s(x, 1, size_x);
    alpha_fill_random_s(y, 1, size_y);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo;
    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_s_mm_plain(transA, alpha, coo, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_s_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_mm");

    alphasparse_destroy(coo);

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

    const float alpha = 3.;
    const float beta = 2.;

    const float mkl_alpha = 3.;
    const float mkl_beta = 2.;

    printf("thread_num : %d\n", thread_num);

    float *alpha_y; 
    float *mkl_y;
    size_t size_alpha_y, size_mkl_y;

    alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);

    int status = 0;
    if (check)
    {
        mkl_mm(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y, &size_mkl_y);
        check_s((float *)mkl_y,size_mkl_y, (float *)alpha_y,size_alpha_y);
        alpha_free(mkl_y);
    }

    alpha_free(alpha_y);
    return status;
}