/**
 * @brief openspblas trsm bsr test
 * @author jaxxxxxo
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

#define mkl_call_noexit(func, message)\
  do{\
    int status = func;\
    if (status != 0)\
    {\
      printf("%s\n", message);\
      switch (status)\
      {\
      case 1:\
        printf("status : not initialized!!!\n");\
        break;\
      case 2:\
        printf("status : alloc failed!!!\n");\
        break;\
      case 3:\
        printf("status : invalid value!!!\n");\
        break;\
      case 4:\
        printf("status : execution failed!!!\n");\
        break;\
      case 5:\
        printf("status : internal error!!!\n");\
        break;\
      case 6:\
        printf("status : not supported!!!\n");\
        break;\
      default:\
        printf("status : status invalid!!!\n");\
        break;\
      }\
      fflush(0);\
    }\
  }while (0);

static void mkl_trsm(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, double **ret, size_t *size)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    double *values;
    mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
    if (m != k)
    {
        printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
        exit(-1);
    }
    MKL_INT columns = args_get_columns(argc, argv, k);

    size_t size_x = k * columns;
    size_t size_y = m * columns;
    double *x = alpha_malloc(size_x * sizeof(double));
    double *y = alpha_malloc(size_y * sizeof(double));
    alpha_fill_random_d(x, 1, size_x);

    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    int ldx = columns, ldy = columns;
    if (layout == SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = k;
        ldy = m;
    }

    mkl_set_num_threads(thread_num);
    sparse_matrix_t cooA, csrA;
    mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_d_trsm(transA, alpha, csrA, descr, layout, x, columns, ldx, y, ldy), "mkl_sparse_d_trsm");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_d_trsm");
    mkl_sparse_destroy(cooA);
    mkl_sparse_destroy(csrA);

    *ret = y;
    *size = size_y;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void mkl_trsm_bsr(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, double **ret, size_t *size, sparse_layout_t b_layout)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    double *values;
    mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

    MKL_INT columns = args_get_columns(argc, argv, k);

    size_t size_x = k * columns;
    size_t size_y = m * columns;
    double *x = alpha_malloc(size_x * sizeof(double));
    double *y = alpha_malloc(size_y * sizeof(double));
    alpha_fill_random_d(x, 1, size_x);


    //TODO block_layout == layout??
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    //TODO mkl_args_get_blockSize()
    const MKL_INT block_size = 4;
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    int ldx = columns, ldy = columns;
    if (layout == SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = k;
        ldy = m;
    }

    mkl_set_num_threads(thread_num);
    sparse_matrix_t cooA, bsrA;
    b_layout = layout;
    if(b_layout == SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        for(int i = 0; i < nnz; i++)
        {
            row_index[i] ++;
            col_index[i] ++;
        }
        mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ONE, m, k, nnz, row_index, col_index, values);
    }
    else
    {
        mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    }

    mkl_sparse_convert_bsr(cooA, block_size, b_layout , SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    
    mkl_call_noexit(mkl_sparse_d_trsm(transA, alpha, bsrA, descr, layout, x, columns, ldx, y, ldy), "mkl_sparse_d_trsm");
    // mkl_sparse_d_trsm(transA, alpha, bsrA, descr, layout, x, columns, ldx, y, ldy);
    ldy=ldx;
    ldx=ldy;

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_d_trsm");
    mkl_sparse_destroy(cooA);
    mkl_sparse_destroy(bsrA);

    *ret = y;
    *size = size_y;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_trsm(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, double **ret, size_t *size, alphasparse_layout_t b_layout)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    double *values;
    alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

    ALPHA_INT columns = args_get_columns(argc, argv, k);

    size_t size_x = k * columns;
    size_t size_y = m * columns;
    double *x = alpha_malloc(size_x * sizeof(double));
    double *y = alpha_malloc(size_y * sizeof(double));
    alpha_fill_random_d(x, 1, size_x);
    // #ifdef DEBUG
    //     printf("columns is %d alpha_trsm called x is:\n",columns);
    //     for(size_t i = 0 ; i < size_x;i++)
    //         printf("x[%ld],%f\n",i, x[i]);
    // #endif
    //TODO block_layout == layout??
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    //TODO mkl_args_get_blockSize()
    const ALPHA_INT block_size = 4;
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    int ldx = columns, ldy = columns;
    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = k;
        ldy = m;
    }
    // kernel中要求block_layout和X、Y的layout相同
    b_layout = layout;
    alpha_set_thread_num(thread_num);

    alphasparse_matrix_t cooA, bsrA;
    alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(cooA, block_size , b_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_d_trsm_plain(transA, alpha, bsrA, descr, layout, x, columns, ldx, y, ldy), "alphasparse_d_trsm_plain");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_trsm_plain");
    alphasparse_destroy(cooA);
    alphasparse_destroy(bsrA);

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

    const double alpha = 2;

    printf("thread_num : %d\n", thread_num);

    double *alpha_y, *mkl_y;
    size_t size_alpha_y, size_mkl_y;

    int status = 0;
    // kernel中要求block_layout和X、Y的layout相同,alpha_trsm的最后一个参数无效
    alpha_trsm(argc, argv, file, thread_num, alpha, &alpha_y, &size_alpha_y, ALPHA_SPARSE_LAYOUT_ROW_MAJOR);

    if (check)
    {
        mkl_trsm_bsr(argc, argv, file, thread_num, alpha, &mkl_y, &size_mkl_y, SPARSE_LAYOUT_ROW_MAJOR);
        //mkl_trsm(argc, argv, file, thread_num, alpha, &mkl_y, &size_mkl_y);
        status = check_d(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
        alpha_free(mkl_y);
    }

    alpha_free(alpha_y);

    // alpha_trsm(argc, argv, file, thread_num, alpha, &alpha_y, &size_alpha_y, ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR);

    // if (check)
    // {
    //     //mkl_trsm_bsr(argc, argv, file, thread_num, alpha, &mkl_y, &size_mkl_y, SPARSE_LAYOUT_COLUMN_MAJOR);
    //     mkl_trsm(argc, argv, file, thread_num, alpha, &mkl_y, &size_mkl_y);
    //     status = check_d(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
    //     alpha_free(mkl_y);
    // }

    // alpha_free(alpha_y);

    return status;
}