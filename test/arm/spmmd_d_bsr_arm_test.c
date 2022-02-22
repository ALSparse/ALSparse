#include <alphasparse.h>
#include <stdio.h>
# define ROW_MAJOR 0
# define COL_MAJOR 1
# define BLOCK_SIZE 4

static void alpha_spmmd_plain(const int argc, const char *argv[], const char *file, int thread_num, double **ret, size_t *ret_size, alphasparse_layout_t block_layout)
{
    alpha_set_thread_num(thread_num);
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_INT rowsA,rowsB,colsA,colsB;
    double *values;

    const char* fileA = args_get_data_fileA(argc,argv);
    const char* fileB = NULL;
    alphasparse_matrix_t cooA, bsrA, cooB, bsrB;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alpha_read_coo_d(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz);
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE){
        rowsA = k;
        colsA = m;
        fileB = args_get_data_fileA(argc,argv);
    }
    else {
        rowsA = m;
        colsA = k;
        fileB = args_get_data_fileB(argc,argv);
    }

    alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooA, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    alpha_read_coo_d(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz);
    rowsB = m;
    colsB = k;

    alphasparse_d_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooB, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrB), "alphasparse_convert_bsr");

    size_t size_C = rowsA * colsB;
    double *C = alpha_malloc(sizeof(double) * size_C);
    ALPHA_INT ldc;
    if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        ldc = rowsA;
    }
    else
    {
        ldc = colsB;
    }
    printf("rowsA %d, colsA %d, rowsB %d, colsB %d ldc %d \n",rowsA,colsA,rowsB,colsB,ldc);
    
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alphasparse_d_spmmd_plain(transA, bsrA, bsrB, layout, C, ldc);
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_spmmd");
    alphasparse_destroy(cooA);
    alphasparse_destroy(bsrA);

    *ret = C;
    *ret_size = size_C;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}


static void alpha_spmmd(const int argc, const char *argv[], const char *file, int thread_num, double **ret, size_t *ret_size, alphasparse_layout_t block_layout)
{
    alpha_set_thread_num(thread_num);
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_INT rowsA,rowsB,colsA,colsB;
    double *values;

    const char* fileA = args_get_data_fileA(argc,argv);
    const char* fileB = NULL;
    alphasparse_matrix_t cooA, bsrA, cooB, bsrB;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alpha_read_coo_d(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz);
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE){
        rowsA = k;
        colsA = m;
        fileB = args_get_data_fileA(argc,argv);
    }
    else {
        rowsA = m;
        colsA = k;
        fileB = args_get_data_fileB(argc,argv);
    }

    alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooA, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    alpha_read_coo_d(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz);
    rowsB = m;
    colsB = k;

    alphasparse_d_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooB, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrB), "alphasparse_convert_bsr");

    size_t size_C = rowsA * colsB;
    double *C = alpha_malloc(sizeof(double) * size_C);
    ALPHA_INT ldc;
    if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        ldc = rowsA;
    }
    else
    {
        ldc = colsB;
    }
    printf("rowsA %d, colsA %d, rowsB %d, colsB %d ldc %d \n",rowsA,colsA,rowsB,colsB,ldc);
    
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alphasparse_d_spmmd(transA, bsrA, bsrB, layout, C, ldc);
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_spmmd");
    alphasparse_destroy(cooA);
    alphasparse_destroy(bsrA);

    *ret = C;
    *ret_size = size_C;
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

    double *alpha_C;
    double *alpha_plain_C;
    size_t size_alpha_plain_C, size_alpha_C;
    printf("thread_num : %d\n", thread_num);

    alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C, ALPHA_SPARSE_LAYOUT_ROW_MAJOR);
    int status = 0;
    { // block row major

        if (check)
        {
            alpha_spmmd_plain(argc, argv, file, thread_num, &alpha_plain_C, &size_alpha_plain_C, ALPHA_SPARSE_LAYOUT_ROW_MAJOR);
            status = check_d((double *)alpha_plain_C,size_alpha_plain_C, (double *)alpha_C,size_alpha_C);
            alpha_free(alpha_plain_C);
        }
    }

    { // block col major
        alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C, ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR);

        if (check)
        {
            alpha_spmmd_plain(argc, argv, file, thread_num, &alpha_plain_C, &size_alpha_plain_C, ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR);
            status = check_d((double *)alpha_plain_C,size_alpha_plain_C, (double *)alpha_C,size_alpha_C);
            alpha_free(alpha_plain_C);
        }
    }

    alpha_free(alpha_C);
    return status;
}
