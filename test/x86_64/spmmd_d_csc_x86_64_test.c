/**
 * @brief openspblas spmmd csc test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

static sparse_status_t alpha_convert_mkl_csc_d(alphasparse_matrix_t src, sparse_matrix_t *dst, sparse_index_base_t base, ALPHA_INT nnz)
{
    spmat_csc_d_t * mat = (spmat_csc_d_t *)src->mat;
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
    sparse_status_t st =  mkl_sparse_d_create_csc(
        dst,
        base,
        mat->rows,
        mat->cols,
        mat->cols_start,
        mat->cols_end,
        mat->row_indx,
        (double*) mat->values
    );
    return st;
}

static void mkl_spmmd(const int argc, const char *argv[], const char *file, int thread_num, double **ret, size_t *ret_size)
{
    // 没有考虑column
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    MKL_INT mA, kA, nnzA, mB, kB, nnzB;
    MKL_INT *row_indexA, *col_indexA, *row_indexB, *col_indexB;
    double *valuesA, *valuesB;
    
    const char *fileA = args_get_data_fileA(argc, argv);
    const char *fileB = NULL;
    mkl_read_coo_d(fileA, &mA, &kA, &nnzA, &row_indexA, &col_indexA, &valuesA);
    if(transA == SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    mkl_read_coo_d(fileB, &mB, &kB, &nnzB, &row_indexB, &col_indexB, &valuesB);

    mkl_set_num_threads(thread_num);

    size_t size_C;
    MKL_INT ldc;
    if(transA == SPARSE_OPERATION_NON_TRANSPOSE){
        size_C = mA * kB;
        if(layout == SPARSE_LAYOUT_COLUMN_MAJOR) ldc = mA;
        else ldc = kB;
    }
    else{
        size_C = kA * kB;
        if(layout == SPARSE_LAYOUT_COLUMN_MAJOR) ldc = kA;
        else ldc = kB;
    }
    
    double *C = alpha_malloc(sizeof(double) * size_C);

    sparse_matrix_t result;
    // create cscA
    alphasparse_matrix_t cooA, alpha_cscA;
    sparse_matrix_t cscA;
    alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, mA, kA, nnzA, row_indexA, col_indexA, valuesA);
    alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
    alpha_convert_mkl_csc_d(alpha_cscA, &cscA, SPARSE_INDEX_BASE_ZERO, nnzA); 
    // mkl_call_exit(mkl_sparse_d_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "mkl_sparse_d_create_coo");
    // mkl_call_exit(mkl_sparse_convert_csc(coo, SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "mkl_sparse_convert_csc");
    
    // create cscB
    alphasparse_matrix_t cooB, alpha_cscB;
    sparse_matrix_t cscB;
    alphasparse_d_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, mB, kB, nnzB, row_indexB, col_indexB, valuesB);
    alphasparse_convert_csc(cooB, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscB);
    alpha_convert_mkl_csc_d(alpha_cscB, &cscB, SPARSE_INDEX_BASE_ZERO, nnzB);

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    mkl_sparse_d_spmmd(transA, cscA, cscB, layout, C, ldc);
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_d_spmmd");

    // save run time
    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }

    alphasparse_destroy(cooA);
    alphasparse_destroy(cooB);
    mkl_sparse_destroy(cscA);
    mkl_sparse_destroy(cscB);

    *ret = C;
    *ret_size = size_C;
    alpha_free(row_indexA);
    alpha_free(col_indexA);
    alpha_free(valuesA);
    alpha_free(row_indexB);
    alpha_free(col_indexB);
    alpha_free(valuesB);
}

static void alpha_spmmd(const int argc, const char *argv[], const char *file, int thread_num, double **ret, size_t *ret_size)
{
    ALPHA_INT mA, kA, nnzA, mB, kB, nnzB;
    ALPHA_INT *row_indexA, *col_indexA, *row_indexB, *col_indexB;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    double *valuesA, *valuesB;
    const char *fileA = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileA, &mA, &kA, &nnzA, &row_indexA, &col_indexA, &valuesA);
    const char *fileB = NULL;
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    alpha_read_coo_d(fileB, &mB, &kB, &nnzB, &row_indexB, &col_indexB, &valuesB);

    size_t size_C;
    ALPHA_INT ldc;
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
        size_C = mA * kB;
        if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) ldc = kB;
        else ldc = kB;
    }
    else{
        size_C = kA * kB;
        if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) ldc = kA;
        else ldc = kA;
    }
    double *C = alpha_malloc(sizeof(double) * size_C);

    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, cscA, cscB, result;

    // create cscA
    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, mA, kA, nnzA, row_indexA, col_indexA, valuesA), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");
    alphasparse_destroy(coo);

    // create cscB
    alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, mB, kB, nnzB, row_indexB, col_indexB, valuesB), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB), "alphasparse_convert_csc");
   
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_d_spmmd_plain(transA, cscA, cscB, layout, C, ldc), "alphasparse_d_spmmd");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_spmmd");

    *ret = C;
    *ret_size = size_C;
    alphasparse_destroy(coo);
    alphasparse_destroy(cscA);
    alphasparse_destroy(cscB);
    alpha_free(row_indexA);
    alpha_free(col_indexA);
    alpha_free(valuesA);
    alpha_free(row_indexB);
    alpha_free(col_indexB);
    alpha_free(valuesB);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = NULL;//args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    double *mkl_C, *alpha_C;
    size_t size_mkl_C, size_alpha_C;

    int status = 0;
    //alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C);

    if (check)
    {
        mkl_spmmd(argc, argv, file, thread_num, &mkl_C, &size_mkl_C);
        //status = check_s(mkl_C, size_mkl_C, alpha_C, size_alpha_C);
        alpha_free(mkl_C);
    }

    //alpha_free(alpha_C);
    return status;
}