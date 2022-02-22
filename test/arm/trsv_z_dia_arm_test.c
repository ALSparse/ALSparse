/**
 * @brief openspblas trsv csr test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <stdio.h>

void plain_trsv(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex16 *values, ALPHA_Complex16 *x, ALPHA_Complex16 alpha, ALPHA_Complex16 *y, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, csrA;
    alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_dia(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "alphasparse_convert_dia");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_trsv_plain(transA, alpha, csrA, descr, x, y), "alphasparse_z_trsv_plain");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_z_trsv_plain");
    alphasparse_destroy(cooA);
    alphasparse_destroy(csrA);
}

void alpha_trsv(const int argc, const char *argv[], ALPHA_INT m, ALPHA_INT n, ALPHA_INT nnz, ALPHA_INT *row_index, ALPHA_INT *col_index, ALPHA_Complex16 *values, ALPHA_Complex16 *x, ALPHA_Complex16 alpha, ALPHA_Complex16 *y, int thread_num)
{
    alpha_set_thread_num(thread_num);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_matrix_t cooA, csrA;
    alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, n, nnz, row_index, col_index, values), "alphasparse_z_create_coo");
    alpha_call_exit(alphasparse_convert_dia(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA), "alphasparse_convert_dia");

    alpha_timer_t timer;
    alpha_timing_start(&timer);

    alpha_call_exit(alphasparse_z_trsv(transA, alpha, csrA, descr, x, y), "alphasparse_z_trsv");

    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_z_trsv");

    // save run time
    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }
    
    alphasparse_destroy(cooA);
    alphasparse_destroy(csrA);
}

int main(int argc,const char *argv[])
{
    // args
    args_help(argc, argv);
    const char *file = args_get_data_file(argc, argv);
    int thread_num = args_get_thread_num(argc, argv);
    bool check = args_get_if_check(argc, argv);

    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    ALPHA_Complex16 *values;
    ALPHA_Complex16 *x, *alpha_y, *plain_y;
    const ALPHA_Complex16 alpha = {.real = 2., .imag = 2.};

    // read coo
    alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

    // init x y
    x = alpha_malloc(k * sizeof(ALPHA_Complex16));
    alpha_y = alpha_malloc(m * sizeof(ALPHA_Complex16));
    plain_y = alpha_malloc(m * sizeof(ALPHA_Complex16));

    // alpha_fill_random_s(x, 0, k);
    alpha_fill_random_z(x, 1, k);

    printf("thread_num : %d\n", thread_num);

    alpha_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, alpha_y, thread_num);

    int status = 0;
    if (check)
    {
        plain_trsv(argc, argv, m, k, nnz, row_index, col_index, values, x, alpha, plain_y, thread_num);
        status = check_z(plain_y, m, alpha_y, m);
    }
    alpha_free(x);
    alpha_free(alpha_y);
    alpha_free(plain_y);

    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}