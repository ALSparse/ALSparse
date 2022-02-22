
/**
 * @brief openspblas mm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <stdio.h>

const char *file;
int thread_num;
bool check;

alphasparse_operation_t transA;
alphasparse_layout_t layout;
struct alpha_matrix_descr descr;
alphasparse_layout_t layout;

ALPHA_INT m, k, nnz;
ALPHA_INT *row_index, *col_index;
float *values;
const float alpha = 2.f;
const float beta = 2.f;

ALPHA_INT ldx, ldy;

ALPHA_INT columns;

float *x;
float *icty;
float *icty_plain;

ALPHA_INT lo, diag, hi;
ALPHA_INT64 ops;

void alpha_mm()
{
    // 设置使用线程数
    alpha_set_thread_num(thread_num);
    alphasparse_matrix_t coo, csr;
    // 创建coo格式稀疏矩阵
    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr), "alphasparse_convert_csr");
    alpha_timer_t timer;
    alpha_timing_start(&timer);
    // 稀疏矩阵乘稠密矩阵
    alpha_call_exit(alphasparse_s_mm(transA, alpha, csr, descr, layout, x, columns, ldx, beta, icty, ldy), "alphasparse_s_mm");
    alpha_timing_end(&timer);
    alpha_timing_elaped_time_print(&timer, "alphasparse_s_mm");
    alpha_timing_gflops_print(&timer, ops, "alphasparse_s_mm");
    // 释放稀疏矩阵
    alphasparse_destroy(coo);
    alphasparse_destroy(csr);
}

void alpha_mm_plain()
{
    alpha_set_thread_num(thread_num);

    alphasparse_matrix_t coo, csr;
    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values), "alphasparse_s_create_coo");
    alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr), "alphasparse_convert_csr");

    alpha_timer_t timer;
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_s_mm_plain(transA, alpha, csr, descr, layout, x, columns, ldx, beta, icty_plain, ldy), "alphasparse_s_mm_plain");
    alpha_timing_end(&timer);

    alpha_timing_elaped_time_print(&timer, "alphasparse_s_mm_plain");
    alpha_timing_gflops_print(&timer, ops, "alphasparse_s_mm_plain");
    alphasparse_destroy(coo);
    alphasparse_destroy(csr);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    file = args_get_data_file(argc, argv);
    thread_num = args_get_thread_num(argc, argv);
    check = args_get_if_check(argc, argv);
    transA = alpha_args_get_transA(argc, argv);
    layout = alpha_args_get_layout(argc, argv);
    descr = alpha_args_get_matrix_descrA(argc, argv);

    layout = alpha_args_get_layout(argc, argv);

    // read coo
    alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

    columns = args_get_columns(argc, argv, k);

    alphasparse_nnz_counter_coo(row_index, col_index, nnz, &lo, &diag, &hi);
    ops = alphasparse_operations_mm(m, k, lo, diag, hi, transA, descr, columns, ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX);

    ldx = columns, ldy = columns;
    ALPHA_INT rowsx = k, rowsy = m;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE || transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        rowsx = m;
        rowsy = k;
    }
    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        ldx = rowsx;
        ldy = rowsy;
    }
    ALPHA_INT sizex = rowsx * columns;
    ALPHA_INT sizey = rowsy * columns;

    x = (float *)alpha_memalign(sizeof(float) * sizex, DEFAULT_ALIGNMENT);
    icty = (float *)alpha_memalign(sizeof(float) * sizey, DEFAULT_ALIGNMENT);
    icty_plain = (float *)alpha_memalign(sizeof(float) * sizey, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(x, 2, sizex);
    alpha_fill_random_s(icty, 1, sizey);
    // printf("thread_num : %d\n",thread_num);
    printf("%d\n", thread_num);
    alpha_mm();
    int status = 0;
    if (check)
    {
        alpha_fill_random_s(icty_plain, 1, sizey);
        alpha_mm_plain();
        status = check_s_l3((float *)icty, ldy, sizey, icty_plain, ldy, sizey, NULL, x, ldx, icty, ldy, alpha, beta, argc, argv);
        // status = check_s(icty, sizey, icty_plain, sizey);
    }
    printf("\n");
    alpha_free(x);
    alpha_free(icty);
    alpha_free(icty_plain);
    alpha_free(row_index);
    alpha_free(col_index);
    alpha_free(values);
    return status;
}
