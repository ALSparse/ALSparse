/**
 * @brief openspblas trsm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>
#include <mkl.h>

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

    size_t size_x, size_y;
    sparse_operation_t transA = mkl_args_get_transA(argc, argv);
    sparse_layout_t layout = mkl_args_get_layout(argc, argv);
    MKL_INT ldx = columns, ldy = columns;
    sparse_index_base_t base_mkl = SPARSE_INDEX_BASE_ZERO;
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
#ifdef DEBUG
    printf("mkl_trsm columns is %d\n",(int)columns);
#endif
    double *x = alpha_malloc(size_x * sizeof(double));
    double *y = alpha_malloc(size_y * sizeof(double));
    alpha_fill_random_d(x, 1, size_x);


    struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

    // int ldx = columns, ldy = columns;
    // if (layout == SPARSE_LAYOUT_COLUMN_MAJOR)
    // {
    //     ldx = k;
    //     ldy = m;
    // }

    mkl_set_num_threads(thread_num);
    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");
    sparse_matrix_t cscB;
    mkl_sparse_d_create_csc(&cscB, base_mkl, m, k, ((spmat_csc_d_t *)cscA->mat)->cols_start, ((spmat_csc_d_t *)cscA->mat)->cols_end, ((spmat_csc_d_t *)cscA->mat)->row_indx, ((spmat_csc_d_t *)cscA->mat)->values);
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_d_trsm(transA, alpha, cscB, descr, layout, x, columns, ldx, y, ldy), "mkl_sparse_d_trsm");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "mkl_sparse_d_trsm");

    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }
    
    alphasparse_destroy(cooA);
    alphasparse_destroy(cscA);
    mkl_sparse_destroy(cscB);

    *ret = y;
    *size = size_y;
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
}

static void alpha_trsm(const int argc, const char *argv[], const char *file, int thread_num, const double alpha, double **ret, size_t *size)
{
    ALPHA_INT m, k, nnz;
    ALPHA_INT *row_index, *col_index;
    double *values;
    alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
    if (m != k)
    {
        printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
        exit(-1);
    }
    ALPHA_INT columns = args_get_columns(argc, argv, k);

    alphasparse_operation_t transA = alpha_args_get_transA(argc, argv);
    alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
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
#ifdef DEBUG
    printf("alpha_trsm columns is %d\n",(int)columns);
#endif
    // size_t size_x = k * columns;
    // size_t size_y = m * columns;
    double *x = alpha_malloc(size_x * sizeof(double));
    double *y = alpha_malloc(size_y * sizeof(double));
    alpha_fill_random_d(x, 1, size_x);

    struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

    // int ldx = columns, ldy = columns;
    // if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    // {
    //     ldx = k;
    //     ldy = m;
    // }
    alpha_set_thread_num(thread_num);

    alphasparse_matrix_t cooA, cscA;
    alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_d_create_coo");
    alpha_call_exit(alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA), "alphasparse_convert_csc");
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_d_trsm_plain(transA, alpha, cscA, descr, layout, x, columns, ldx, y, ldy), "alphasparse_d_trsm_plain");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_d_trsm_plain");
    FILE* fp;
    fp = fopen("1.txt", "a+");
    if (fp) {
        const alpha_timer_t* ttt = &timer;
        fprintf(fp, "%lf\n", alpha_timing_elapsed_time(ttt));
        fclose(fp);
    }

    alphasparse_destroy(cooA);
    alphasparse_destroy(cscA);

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
    //alpha_trsm(argc, argv, file, thread_num, alpha, &alpha_y, &size_alpha_y);

    if (check)
    {
        mkl_trsm(argc, argv, file, thread_num, alpha, &mkl_y, &size_mkl_y);
        //status = check_d(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
        alpha_free(mkl_y);
    }
#ifdef DEBUG
//    printf("alpha_trsm result is:\n ");
//    for(size_t i = 0 ; i < size_alpha_y; i++){
//        printf("%f%c",alpha_y[i],i%10==0?'\n':',');
//    }
//    printf("\n");
//
//    printf("mklt_trsm result is:\n ");
//    for(size_t i = 0 ; i < size_alpha_y; i++){
//        printf("%f%c",mkl_y[i],i%10==0?'\n':',');
//    }
//    printf("\n");
//    printf("dist is:\n ");
//    for(size_t i = 0 ; i < size_alpha_y; i++){
//        printf("%f%c",alpha_y[i]-mkl_y[i],i%10==0?'\n':',');
//    }
//    printf("\n");
#endif

    //alpha_free(alpha_y);
    return status;
}
