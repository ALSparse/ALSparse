/**
 * @brief openspblas spmm bsr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
# define ROW_MAJOR 0
# define COL_MAJOR 1
# define BLOCK_SIZE 4

static void alpha_spmm_plain(const int argc, const char *argv[], int thread_num, 
                            alphasparse_index_base_t *ret_index, 
                            ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, 
                            ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, 
                            ALPHA_Complex16 **ret_values, 
                            int flag)
{
    alpha_set_thread_num(thread_num);
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_INT rowsA,rowsB,colsA,colsB;
    ALPHA_Complex16 *values;
    const char* fileA = args_get_data_fileA(argc,argv);
    const char* fileB = NULL;
    alphasparse_matrix_t cooA, bsrA, cooB, bsrB,result;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = ALPHA_SPARSE_LAYOUT_ROW_MAJOR;
    
    alpha_read_coo_z(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz * 2);
    // alpha_set_num_threads(thread_num);
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

    alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooA, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    alpha_read_coo_z(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz * 2);
    rowsB = m;
    colsB = k;
    printf("rowsA %d, colsA %d, rowsB %d, colsB %d\n",rowsA,colsA,rowsB,colsB);

    alphasparse_z_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooB, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrB), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_spmm_plain(transA, bsrA, bsrB, &result), "alphasparse_spmm");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_spmm");

    // alphasparse_order(result);

    alphasparse_layout_t alpha_block_layout;
    ALPHA_INT alpha_block_size;
    alpha_call_exit(alphasparse_z_export_bsr(result, ret_index, &alpha_block_layout, ret_rows, ret_cols, &alpha_block_size, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_z_export_bsr");

    alphasparse_destroy(cooA);
    alphasparse_destroy(cooB);
    alphasparse_destroy(result);
    
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}


static void alpha_spmm( const int argc, const char *argv[], int thread_num, 
                            alphasparse_index_base_t *ret_index, 
                            ALPHA_INT *ret_rows, ALPHA_INT *ret_cols, 
                            ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index, 
                            ALPHA_Complex16 **ret_values, 
                            int flag )
{
    alpha_set_thread_num(thread_num);
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_INT rowsA,rowsB,colsA,colsB;
    ALPHA_Complex16 *values;
    const char* fileA = args_get_data_fileA(argc,argv);
    const char* fileB = NULL;
    alphasparse_matrix_t cooA, bsrA, cooB, bsrB,result;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = ALPHA_SPARSE_LAYOUT_ROW_MAJOR;
    
    alpha_read_coo_z(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz * 2);
    // alpha_set_num_threads(thread_num);
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

    alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooA, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA), "alphasparse_convert_bsr");
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);

    alpha_read_coo_z(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
    alpha_fill_random_d((double *)values, 1, nnz * 2);
    rowsB = m;
    colsB = k;
    printf("rowsA %d, colsA %d, rowsB %d, colsB %d\n",rowsA,colsA,rowsB,colsB);

    alphasparse_z_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
    alpha_call_exit(alphasparse_convert_bsr(cooB, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrB), "alphasparse_convert_bsr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_spmm(transA, bsrA, bsrB, &result), "alphasparse_spmm");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_spmm");

    // alphasparse_order(result);

    alphasparse_layout_t alpha_block_layout;
    ALPHA_INT alpha_block_size;
    alpha_call_exit(alphasparse_z_export_bsr(result, ret_index, &alpha_block_layout, ret_rows, ret_cols, &alpha_block_size, ret_rows_start, ret_rows_end, ret_col_index, ret_values), "alphasparse_z_export_bsr");

    alphasparse_destroy(cooA);
    alphasparse_destroy(cooB);
    alphasparse_destroy(result);
    
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

    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex16 *values;

    // return
    int status = 0;
    int flag = ROW_MAJOR;
    alphasparse_index_base_t alpha_plain_index;
    ALPHA_INT alpha_plain_rows, alpha_plain_cols, *alpha_plain_rows_start, *alpha_plain_rows_end, *alpha_plain_col_index;
    ALPHA_Complex16 *alpha_plain_values;

    alphasparse_index_base_t alpha_index;
    ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
    ALPHA_Complex16 *alpha_values;

    alpha_spmm(argc, argv, thread_num, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values, flag);

    if (check)
    {
        alpha_spmm_plain(argc, argv, thread_num, &alpha_plain_index, &alpha_plain_rows, &alpha_plain_cols, &alpha_plain_rows_start, &alpha_plain_rows_end, &alpha_plain_col_index, &alpha_plain_values, flag);
        int alpha_plain_nnz = alpha_plain_rows_end[alpha_plain_rows/BLOCK_SIZE - 1];
        int alpha_nnz = alpha_rows_end[alpha_rows/BLOCK_SIZE - 1];

        status = check_z(alpha_plain_values, alpha_plain_nnz*BLOCK_SIZE*BLOCK_SIZE, alpha_values, alpha_nnz*BLOCK_SIZE*BLOCK_SIZE);
    }

    return status;
}
