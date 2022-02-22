/**
 * @brief openspblas spmmd csc test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>

static void alpha_spmmd(const int argc, const char *argv[], ALPHA_INT mA, ALPHA_INT nA, ALPHA_INT mB, ALPHA_INT nB, \
    ALPHA_INT nnzA, ALPHA_INT *row_indexA, ALPHA_INT *col_indexA, float *valuesA, \
    ALPHA_INT nnzB, ALPHA_INT *row_indexB, ALPHA_INT *col_indexB, float *valuesB, \
    float *C, ALPHA_INT ldc, ALPHA_INT thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, mA, nA, nnzA, row_indexA, col_indexA, valuesA), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");

    alphasparse_matrix_t cooB, cscB;
    alpha_call_exit(alphasparse_s_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, mB, nB, nnzB, row_indexB, col_indexB, valuesB), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooB, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_spmmd(transA, cscA, cscB, layout, C, ldc), "alphasparse_s_spmmd");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_spmmd");
    alphasparse_destroy(cooA);
    alphasparse_destroy(cscA);
    alphasparse_destroy(cooB);
    alphasparse_destroy(cscB);
}

static void alpha_spmmd_plain(const int argc, const char *argv[], ALPHA_INT mA, ALPHA_INT nA, ALPHA_INT mB, ALPHA_INT nB, \
    ALPHA_INT nnzA, ALPHA_INT *row_indexA, ALPHA_INT *col_indexA, float *valuesA, \
    ALPHA_INT nnzB, ALPHA_INT *row_indexB, ALPHA_INT *col_indexB, float *valuesB, \
    float *C, ALPHA_INT ldc, ALPHA_INT thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, mA, nA, nnzA, row_indexA, col_indexA, valuesA), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");

    alphasparse_matrix_t cooB, cscB;
    alpha_call_exit(alphasparse_s_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, mB, nB, nnzB, row_indexB, col_indexB, valuesB), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooB, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB), "alphasparse_convert_csc");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_s_spmmd_plain(transA, cscA, cscB, layout, C, ldc), "alphasparse_s_spmmd_plain");

    alpha_timing_end(&timer);

    alpha_timing_elaped_time_print(&timer, "alphasparse_s_spmmd_plain");
    alphasparse_destroy(cooA);
    alphasparse_destroy(cscA);
    alphasparse_destroy(cooB);
    alphasparse_destroy(cscB);
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *fileA = args_get_data_fileA(argc, argv);
    const char *fileB = NULL;
    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    if(transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        fileB = args_get_data_fileB(argc, argv);
    else
        fileB = args_get_data_fileA(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    printf("thread num : %d\n", thread_num);

    ALPHA_INT mA, nA, mB, nB, nnzA, nnzB;
    ALPHA_INT *row_indexA, *col_indexA, *row_indexB, *col_indexB;
    float *valuesA, *valuesB;

    // read coo
    alpha_read_coo(fileA, &mA, &nA, &nnzA, &row_indexA, &col_indexA, &valuesA);
    alpha_read_coo(fileB, &mB, &nB, &nnzB, &row_indexB, &col_indexB, &valuesB);
    ALPHA_INT C_m, C_n, ldc;
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
    if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE){
        C_m = nA;
        C_n = nA;
        ldc = nA;
    }
    else{
        C_m = mA;
        C_n = nB;
        if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
            ldc = nB;
        else
            ldc = nB;
    }

    float *alpha_C = alpha_malloc(sizeof(float) * (C_m * C_n));
    float *alpha_C_plain = alpha_malloc(sizeof(float) * (C_m * C_n));

    alpha_fill_random_s(valuesA, 0, nnzA);
    alpha_fill_random_s(valuesB, 0, nnzB);

    alpha_spmmd(argc, argv, mA, nA, mB, nB, \
        nnzA, row_indexA, col_indexA, valuesA, \
        nnzB, row_indexB, col_indexB, valuesB, \
        alpha_C, ldc, thread_num);
    int status = 0;
    if (check)
    {
        alpha_spmmd_plain(argc, argv, mA, nA, mB, nB, \
            nnzA, row_indexA, col_indexA, valuesA,\
            nnzB, row_indexB, col_indexB, valuesB,\
            alpha_C_plain, ldc, thread_num);
        status = check_s(alpha_C, C_m * C_n, alpha_C_plain, C_m * C_n);
    }

    alpha_free(alpha_C);
    alpha_free(alpha_C_plain);

    alpha_free(row_indexA);
    alpha_free(col_indexA);
    alpha_free(valuesA);

    alpha_free(row_indexB);
    alpha_free(col_indexB);
    alpha_free(valuesB);
    return status;
}