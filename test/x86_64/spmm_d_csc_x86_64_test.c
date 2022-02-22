/**
 * @brief openspblas spmm csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static sparse_status_t alpha_convert_mkl_csc_d(alphasparse_matrix_t src, sparse_matrix_t *dst)
{
    spmat_csc_d_t * mat = (spmat_csc_d_t *)src->mat;
    sparse_status_t st =  mkl_sparse_d_create_csc(
        dst,
        SPARSE_INDEX_BASE_ZERO,
        mat->rows,
        mat->cols,
        mat->cols_start,
        mat->cols_end,
        mat->row_indx,
        (double*) mat->values
    );
    return st;
}

static void mkl_spmm(const int argc, const char *argv[], const char *file, int thread_num, sparse_index_base_t *ret_index, MKL_INT *ret_rows, MKL_INT *ret_cols, MKL_INT **ret_rows_start, MKL_INT **ret_rows_end, MKL_INT **ret_col_index, double **ret_values)
{
    MKL_INT m, k, nnz;
    MKL_INT *row_index, *col_index;
    double *values;
    // mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
    // alpha_fill_random_s(values, 1, nnz);

    // mkl_set_num_threads(thread_num);
    // sparse_operation_t transA = mkl_args_get_transA(argc, argv);

    // sparse_matrix_t coo, csr, result;
    // mkl_call_exit(mkl_sparse_d_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "mkl_sparse_d_create_coo");
    // if(transA == SPARSE_OPERATION_NON_TRANSPOSE)
    //     {mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_TRANSPOSE, &csr), "mkl_sparse_convert_csr");}
    // else
    //     mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr), "mkl_sparse_convert_csr");
    
    sparse_matrix_t result;
    // create cscA
    const char *fileA = args_get_data_fileA(argc, argv);
    mkl_read_coo_d(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

    mkl_set_num_threads(thread_num);
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);

    // sparse_matrix_t coo, csrA, csrB, result;
    // mkl_call_exit(mkl_sparse_d_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "mkl_sparse_d_create_coo");
    // mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "mkl_sparse_convert_csr");
    alphasparse_matrix_t cooA, alpha_cscA;
    sparse_matrix_t cscA;
    alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
    alpha_convert_mkl_csc_d(alpha_cscA, &cscA); 
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    // create cscB
    const char *fileB = NULL;
    if(transA == SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    mkl_read_coo_d(fileB, &m, &k, &nnz, &row_index, &col_index, &values);

    // mkl_call_exit(mkl_sparse_d_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "mkl_sparse_d_create_coo");
    // mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csrB), "mkl_sparse_convert_csr");
    alphasparse_matrix_t cooB, alpha_cscB;
    sparse_matrix_t cscB;
    alphasparse_d_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alphasparse_convert_csc(cooB, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscB);
    alpha_convert_mkl_csc_d(alpha_cscB, &cscB);

    // spmm
    alpha_timer_t timer;
    alpha_timing_start(&timer);

    mkl_call_exit(mkl_sparse_spmm(transA, cscA, cscB, &result), "mkl_sparse_spmm");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_spmm");

    // save run time
    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }

    mkl_sparse_order(result);

    mkl_call_exit(mkl_sparse_d_export_csc(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "mkl_sparse_d_export_csc");

    alphasparse_destroy(cooA);
    alphasparse_destroy(cooB);
    mkl_sparse_destroy(cscA);
    mkl_sparse_destroy(cscB);
    
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_spmm(const int argc, const char *argv[], const char *file, int thread_num, alphasparse_index_base_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, double **ret_values)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    double *values;
    const char *fileA = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
    //alpha_fill_random_s(values, 1, nnz);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, cscA, cscB, result;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);

    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");
    alphasparse_destroy(coo);   
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    const char *fileB = NULL;
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_spmm_plain(transA, cscA, cscB, &result), "alphasparse_spmm_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_spmm_plain");

    alpha_call_exit(alphasparse_d_export_csc(result, ret_index, ret_rows, ret_cols, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_d_export_csc");

    alphasparse_destroy(coo);
    alphasparse_destroy(cscA);
    alphasparse_destroy(cscB);
    
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    // return
    sparse_index_base_t mkl_index;
    MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index;
    double *mkl_values;

    alphasparse_index_base_t alpha_index;
    ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
    double *alpha_values;

    //alpha_spmm(argc, argv, file, thread_num, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values);

    int status = 0;
    if (check)
    {
        mkl_spmm(argc, argv, file, thread_num, &mkl_index, &mkl_rows, &mkl_cols, &mkl_rows_start, &mkl_rows_end, &mkl_col_index, &mkl_values);
        //int mkl_nnz = mkl_rows_end[mkl_rows - 1];
        //int alpha_nnz = alpha_rows_end[alpha_rows - 1];
        //status = check_s(mkl_values, mkl_nnz, alpha_values, alpha_nnz);
    }

    return status;
}