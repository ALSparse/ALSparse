/**
 * @brief openspblas mv bsr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>
 
static int iter;

static void mkl_mv(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, const double beta, double **ret_y, size_t *ret_size_y)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    double *values;
    mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

    size_t size_x = k;
    size_t size_y = m;
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    if(transA == SPARSE_OPERATION_TRANSPOSE){
        size_x = m;
        size_y = k;
    }
    double *x = alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
    double *y = alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_d(values, 1, nnz);
    alpha_fill_random_d(x, 1, size_x);
    alpha_fill_random_d(y, 1, size_y);

    mkl_set_num_threads(thread_num);
    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    sparse_matrix_t cooA, bsrA;
    mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    mkl_sparse_convert_bsr(cooA, 4, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);

    alpha_timer_t timer;
    //alpha_timing_start(&timer);

    double total_time = 0.;    
	for(int i = 0;i<iter;i++){    
		alpha_clear_cache();    
		alpha_timing_start(&timer);    
		mkl_call_exit(mkl_sparse_d_mv(transA, alpha, bsrA, descr, x, beta, y), "mkl_sparse_d_mv");    
		alpha_timing_end(&timer);    
		total_time += alpha_timing_elapsed_time(&timer);    
	}    
	printf("iter is %d, %s avg time : %lf[sec]\n",iter,"mkl_sparse_d_mv",total_time/iter);

    //alpha_timing_end(&timer);


    mkl_sparse_destroy(cooA);
    mkl_sparse_destroy(bsrA);

    *ret_y = y;
    *ret_size_y = size_y;

    alpha_free(x);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}
static void alpha_mv(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, const double beta, double **ret_y, size_t *ret_size_y)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    double *values;
    alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

    size_t size_x = k;
    size_t size_y = m;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE){
        size_x = m;
        size_y = k;
    }
    double *x = alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
    double *y = alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

    alpha_fill_random_d(values, 1, nnz);
    alpha_fill_random_d(x, 1, size_x);
    alpha_fill_random_d(y, 1, size_y);

    alpha_set_thread_num(thread_num);

    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, bsrA;
    alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_bsr(cooA, 4, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    //alpha_timing_start(&timer);

    double total_time = 0.;    
	for(int i = 0;i<iter;i++){    
		alpha_clear_cache();    
		alpha_timing_start(&timer);    
		alpha_call_exit(alphasparse_d_mv_plain(transA, alpha, bsrA, descr, x, beta, y), "alphasparse_d_mv");    
		alpha_timing_end(&timer);    
		total_time += alpha_timing_elapsed_time(&timer);    
	}    
	printf("iter is %d, %s avg time : %lf[sec]\n",iter,"alphasparse_d_mv_plain",total_time/iter);;

    //alpha_timing_end(&timer);
    alphasparse_destroy(cooA);
    alphasparse_destroy(bsrA);

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
    iter = args_get_iter(argc,argv);

    const double alpha = 2;
    const double beta = 3;

    double *alpha_y, *mkl_y;
    size_t size_alpha_y, size_mkl_y;

    printf("thread_num : %d\n", thread_num);

    alpha_mv(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y); 

    int status = 0;

    if (check)
    {
        mkl_mv(argc, argv, file, thread_num, alpha, beta, &mkl_y, &size_mkl_y);
        
        status = check_d(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
        alpha_free(mkl_y);
    }

    alpha_free(alpha_y);
    return status;
}