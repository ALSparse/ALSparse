/**
 * @brief openspblas trsv csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static sparse_status_t alpha_convert_mkl_csc_s(alphasparse_matrix_t src, sparse_matrix_t *dst)
{
    spmat_csc_s_t * mat = (spmat_csc_s_t *)src->mat;
    sparse_status_t st =  mkl_sparse_s_create_csc(
        dst,
        SPARSE_INDEX_BASE_ZERO,
        mat->rows,
        mat->cols,
        mat->cols_start,
        mat->cols_end,
        mat->row_indx,
        (float*) mat->values
    );
    return st;
}

static void mkl_trsv(const int argc, const char *argv[], const char *file, int thread_num, const float alpha, float **ret, size_t *size)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    float *values;
    mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
    if (m != k)
    {
        printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
        exit(-1);
    }
    size_t size_x = k;
    size_t size_y = m;
    float *x = alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
    float *y = alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);
    alpha_fill_random_s(x, 1, size_x);

    mkl_set_num_threads(thread_num);

    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    // sparse_matrix_t cooA, csrA;
    // mkl_sparse_s_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    // mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
    alpha_timer_t timer;
    alphasparse_matrix_t cooA, alpha_cscA;
    sparse_matrix_t cscA;
    alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
    alpha_convert_mkl_csc_s(alpha_cscA, &cscA);

    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_s_trsv(transA, alpha, cscA, descr, x, y), "mkl_sparse_s_trsv");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_s_trsv");

    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }

    mkl_sparse_destroy(cscA);
    alphasparse_destroy(cooA);
    alphasparse_destroy(alpha_cscA);

    *ret = y;
    *size = size_y;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_trsv(const int argc, const char *argv[], const char *file, int thread_num, const float alpha, float **ret, size_t *size)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    float *values;
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
    if (m != k)
    {
        printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
        exit(-1);
    }
    if (m != k)
    {
        printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
        exit(-1);
    }
    size_t size_x = k;
    size_t size_y = m;
    float *x = alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
    float *y = alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);
    alpha_fill_random_s(x, 1, size_x);

    alpha_set_thread_num(thread_num);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, csrA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_trsv_plain(transA, alpha, csrA, descr, x, y), "alphasparse_s_trsv_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_trsv_plain");
    alphasparse_destroy(cooA);
    alphasparse_destroy(csrA);

    *ret = y;
    *size = size_y;
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

    const float alpha = 2;

    printf("thread_num : %d\n", thread_num);

    float *alpha_y, *mkl_y;
    size_t size_alpha_y, size_mkl_y;

    //alpha_trsv(argc, argv, file, thread_num, alpha, &alpha_y, &size_alpha_y);
    int status = 0;
    if (check)
    {
        mkl_trsv(argc, argv, file, thread_num, alpha, &mkl_y, &size_mkl_y);
        //status = check_s(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
        alpha_free(mkl_y);
    }

    //alpha_free(alpha_y);

    return status;
}