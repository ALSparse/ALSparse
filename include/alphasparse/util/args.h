#pragma once

#include <getopt.h>
#include <stdlib.h>
#include "alphasparse/spdef.h"
#include <stdbool.h>

#define DEFAULT_DATA_FILE ""
#define DEFAULT_THREAD_NUM 1
#define DEFAULT_CHECK false
#define DEFAULT_LAYOUT ALPHA_SPARSE_LAYOUT_ROW_MAJOR
#define DEFAULT_SPARSE_OPERATION ALPHA_SPARSE_OPERATION_NON_TRANSPOSE
#define DEFAULT_ITER 1

alphasparse_layout_t alphasparse_layout_parse(const char *arg);
alphasparse_operation_t alphasparse_operation_parse(const char *arg);
alphasparse_matrix_type_t alphasparse_matrix_type_parse(const char *arg);
alphasparse_fill_mode_t alphasparse_fill_mode_parse(const char *arg);
alphasparse_diag_type_t alphasparse_diag_type_parse(const char *arg);

void args_help(const int argc, const char *argv[]);
bool args_get_if_check(const int argc, const char *argv[]);
int args_get_thread_num(const int argc, const char *argv[]);
int args_get_columns(const int argc, const char *argv[], int k);
int args_get_iter(const int argc, const char *argv[]);

const char* args_get_data_file(const int argc, const char *argv[]);
const char* args_get_data_fileA(const int argc, const char *argv[]);
const char* args_get_data_fileB(const int argc, const char *argv[]);

alphasparse_layout_t alpha_args_get_layout_helper(const int argc, const char *argv[], const int layout_opt);
struct alpha_matrix_descr alpha_args_get_matrix_descr_helper(const int argc, const char *argv[], const int type_opt, const int fill_opt, const int diag_opt);
alphasparse_operation_t alpha_args_get_trans_helper(const int argc, const char *argv[], const int trans_opt);

alphasparse_layout_t alpha_args_get_layout(const int argc, const char *argv[]);
alphasparse_layout_t alpha_args_get_layoutB(const int argc, const char *argv[]);
alphasparse_layout_t alpha_args_get_layoutC(const int argc, const char *argv[]);

struct alpha_matrix_descr alpha_args_get_matrix_descrA(const int argc, const char *argv[]);
struct alpha_matrix_descr alpha_args_get_matrix_descrB(const int argc, const char *argv[]);

alphasparse_operation_t alpha_args_get_transA(const int argc, const char *argv[]);
alphasparse_operation_t alpha_args_get_transB(const int argc, const char *argv[]);

void alpha_arg_parse(const int argc, const char *argv[], alphasparse_layout_t *layout, alphasparse_operation_t *transA, alphasparse_operation_t *transB, struct alpha_matrix_descr *descr);

#ifdef __MKL__
#include <mkl.h>

sparse_layout_t mkl_sparse_layout_parse(const char *arg);
sparse_operation_t mkl_sparse_operation_parse(const char *arg);
sparse_matrix_type_t mkl_sparse_matrix_type_parse(const char *arg);
sparse_fill_mode_t mkl_sparse_fill_mode_parse(const char *arg);
sparse_diag_type_t mkl_sparse_diag_type_parse(const char *arg);

sparse_layout_t mkl_args_get_layout_helper(const int argc, const char *argv[], const int layout_opt);
struct matrix_descr mkl_args_get_matrix_descr_helper(const int argc, const char *argv[], const int type_opt, const int fill_opt, const int diag_opt);
sparse_operation_t mkl_args_get_trans_helper(const int argc, const char *argv[], const int trans_opt);

sparse_layout_t mkl_args_get_layout(const int argc, const char *argv[]);
sparse_layout_t mkl_args_get_layoutB(const int argc, const char *argv[]);
sparse_layout_t mkl_args_get_layoutC(const int argc, const char *argv[]);

struct matrix_descr mkl_args_get_matrix_descrA(const int argc, const char *argv[]);
struct matrix_descr mkl_args_get_matrix_descrB(const int argc, const char *argv[]);

sparse_operation_t mkl_args_get_transA(const int argc, const char *argv[]);
sparse_operation_t mkl_args_get_transB(const int argc, const char *argv[]);
#endif
