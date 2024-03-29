/**
 * @brief openspblas mm csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static sparse_status_t alpha_convert_mkl_csc_s(alphasparse_matrix_t src, sparse_matrix_t *dst, sparse_index_base_t base, ALPHA_INT nnz)
{
    spmat_csc_s_t * mat = (spmat_csc_s_t *)src->mat;
    // if(base == SPARSE_INDEX_BASE_ONE){
    //     for (ALPHA_INT i = 0; i < mat->rows; i++)
    //     {
    //         mat->cols_end[i] = mat->cols_end[i] + 1;
    //     }
    //     for (ALPHA_INT i = 0; nnz; i++) //nnz
    //     {
    //         mat->row_indx[i] = mat->row_indx[i] + 1;
    //     }
    // }
    sparse_status_t st =  mkl_sparse_s_create_csc(
        dst,
        base,
        mat->rows,
        mat->cols,
        mat->cols_start,
        mat->cols_end,
        mat->row_indx,
        (float*) mat->values
    );
    return st;
}

static void mkl_mm(const int argc, const char *argv[], const char *file, int thread_num, float alpha, float beta, float **ret_y, size_t *ret_size_y)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    float *values;
    mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

    MKL_INT columns = args_get_columns(argc, argv, k);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    sparse_index_base_t base_mkl = SPARSE_INDEX_BASE_ZERO;
    size_t size_x, size_y;
    MKL_INT ldx = columns, ldy = columns;
    if(transA == SPARSE_OPERATION_NON_TRANSPOSE){
        size_x = k * columns;
        size_y = m * columns;
        if(layout == SPARSE_LAYOUT_COLUMN_MAJOR){
            ldx = k;
            ldy = m;
            base_mkl = SPARSE_INDEX_BASE_ONE;
        }
    }
    else{
        size_x = m * columns;
        size_y = k * columns;
        if(layout == SPARSE_LAYOUT_COLUMN_MAJOR){
            ldx = m;
            ldy = k;
            base_mkl = SPARSE_INDEX_BASE_ONE;
        }
    }

    float *x = alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
    float *y = alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(values, 1, nnz);
    alpha_fill_random_s(x, 1, size_x);
    alpha_fill_random_s(y, 1, size_y);

    mkl_set_num_threads(thread_num);
    alphasparse_matrix_t cooA, alpha_cscA;
    sparse_matrix_t cscA;
    alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
    alpha_convert_mkl_csc_s(alpha_cscA, &cscA, base_mkl, nnz);
    
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_call_exit(mkl_sparse_s_mm(transA, alpha, cscA, descr, layout, x, columns, ldx, beta, y, ldy), "mkl_sparse_s_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_s_mm");

    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }

    *ret_y = y;
    *ret_size_y = size_y;

    alphasparse_destroy(cooA);
    alphasparse_destroy(alpha_cscA);
    mkl_sparse_destroy(cscA);
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

    size_t size_x, size_y;
    ALPHA_INT ldx = columns, ldy = columns;
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
        size_x = k * columns;
        size_y = m * columns;
        if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
            ldx = k;
            ldy = m;
        }
    }
    else{
        size_x = m * columns;
        size_y = k * columns;
        if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
            ldx = m;
            ldy = k;
        }
    }
    
    float *x = alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
    float *y = alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(values, 1, nnz);
    alpha_fill_random_s(x, 1, size_x);
    alpha_fill_random_s(y, 1, size_y);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csc;
    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csc), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_mm_plain(transA, alpha, csc, descr, layout, x, columns, ldx, beta, y, ldy), "alphasparse_s_mm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_mm");

    alphasparse_destroy(coo);
    alphasparse_destroy(csc);

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

    printf("thread_num : %d\n", thread_num);

    float *alpha_y, *mkl_y;
    size_t size_alpha_y, size_mkl_y;

    //alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);

    int status = 0;
    if (check)
    {
        mkl_mm(argc, argv, file, thread_num, alpha, beta, &mkl_y, &size_mkl_y);
        //for(int i=0; i<10; i++) printf("%f ", (float)mkl_y[i]); //有输出结果，表明不是空指针
        //check_s(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
        //alpha_free(mkl_y); //报错：double free or corruption (out)，太特么玄学了
    }

    //alpha_free(alpha_y);
    return status;
}