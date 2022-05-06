/**
 * @brief ict dcu mm csr test
 * @author HPCRC, ICT
 */

#include <hip/hip_runtime_api.h>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "rocsparse.h"
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <alphasparse_dcu.h>

const char *file;
bool check;

alphasparse_operation_t transA, transB;
rocsparse_operation roctransA, roctransB;
struct alpha_matrix_descr descr;

ALPHA_INT columns;
// csr format
ALPHA_INT A_rows, A_cols, rnnz;
ALPHA_INT *csr_row_ptr, *csr_row_ptr_end, *csr_col_index;
float *csr_values;

// parms for kernel
float *matB_roc, *matB_ict;
ALPHA_INT B_rows, B_cols;
ALPHA_INT ldb;
const float alpha = 2.f;

const ALPHA_INT warm_up = 5;
const ALPHA_INT trials  = 10;
const int batch_size    = 1;

static void roc_trsm_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    // std:: cout << "Device: " << devProp.name << std:: endl;

    rocsparse_int m    = A_rows;
    rocsparse_int nrhs = columns;
    rocsparse_int nnz  = rnnz;
    rocsparse_int ldb_ = ldb;

    // Generate problem
    std::vector<rocsparse_int> hAptr(m + 1);
    std::vector<rocsparse_int> hAcol(nnz);
    std::vector<float> hAval(nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    rocsparse_int *dAptr = NULL;
    rocsparse_int *dAcol = NULL;
    float *dAval         = NULL;
    float *dmatB         = NULL;

    hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void **)&dAval, sizeof(float) * nnz);
    hipMalloc((void **)&dmatB, sizeof(float) * B_cols * ldb);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dmatB, matB_roc, sizeof(float) * B_cols * ldb, hipMemcpyHostToDevice);

    float halpha = alpha;

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);
    if (descr.diag == ALPHA_SPARSE_DIAG_UNIT)
        rocsparse_set_mat_diag_type(descrA, rocsparse_diag_type_unit);
    else
        rocsparse_set_mat_diag_type(descrA, rocsparse_diag_type_non_unit);

    if (descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
        rocsparse_set_mat_fill_mode(descrA, rocsparse_fill_mode_lower);
    else
        rocsparse_set_mat_fill_mode(descrA, rocsparse_fill_mode_upper);

    if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        rocsparse_set_mat_type(descrA, rocsparse_matrix_type_general);
    else {
        std::cout << "type not support!" << std::endl;
        exit(0);
    }

    // Create matrix info structure
    rocsparse_mat_info info;
    rocsparse_create_mat_info(&info);

    double time1 = get_time_us();

    // Obtain required buffer size
    size_t buffer_size;
    rocsparse_scsrsm_buffer_size(
        handle, roctransA, roctransB, m, nrhs, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb_, info, rocsparse_solve_policy_auto, &buffer_size);

    // Allocate temporary buffer
    void *temp_buffer;
    hipMalloc(&temp_buffer, buffer_size);

    // Perform analysis step
    rocsparse_scsrsm_analysis(handle, roctransA, roctransB, m, nrhs, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb_, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, temp_buffer);
    time1 = (get_time_us() - time1) / (trials * batch_size * 1e3);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrsm
        roc_call_exit(rocsparse_scsrsm_solve(
                          handle, roctransA, roctransB, m, nrhs, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb_, info, rocsparse_solve_policy_auto, temp_buffer),
                      "rocsparse_scsrsm_solve");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time2 = get_time_us();

    // cout << "m:" << m << " nrhs:" << nrhs << " nnz:" << nnz << endl;
    // cout << "ldb:" << ldb << endl;
    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call rocsparse csrsm
            roc_call_exit(rocsparse_scsrsm_solve(
                              handle, roctransA, roctransB, m, nrhs, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb_, info, rocsparse_solve_policy_auto, temp_buffer),
                          "rocsparse_scsrsm_solve");
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time2 = (get_time_us() - time2) / (trials * batch_size * 1e3);
    std::cout << time1 + time2 << std::endl;

    hipMemcpy(matB_roc, dmatB, sizeof(float) * B_cols * ldb, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dmatB);

    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);
}

static void alpha_trsm_dcu()
{
    // rocSPARSE handle
    alphasparse_dcu_handle_t handle;
    init_handle(&handle);
    alphasparse_dcu_get_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    // std:: cout << "Device: " << devProp.name << std:: endl;

    // Generate problem
    ALPHA_INT m    = A_rows;
    ALPHA_INT nrhs = columns;
    ALPHA_INT nnz  = rnnz;

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
    ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz);
    float *hAval     = (float *)alpha_malloc(sizeof(float) * nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    ALPHA_INT *dAptr = NULL;
    ALPHA_INT *dAcol = NULL;
    float *dAval     = NULL;
    float *dmatB     = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dmatB, sizeof(float) * B_cols * ldb));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAval, hAval, sizeof(float) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dmatB, matB_ict, sizeof(float) * B_cols * ldb, hipMemcpyHostToDevice));

    float halpha = alpha;

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrA;
    alphasparse_dcu_create_mat_descr(&descrA);
    descrA->diag = descr.diag;
    descrA->mode = descr.mode;
    descrA->type = descr.type;

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu csrsm
        alpha_call_exit(alphasparse_dcu_s_csrsm_solve(
                            handle, transA, transB, m, nrhs, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb, nullptr, ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO, nullptr),
                        "alphasparse_dcu_s_csrsm_solve");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call alphasparse_dcu csrsm
            alpha_call_exit(alphasparse_dcu_s_csrsm_solve(
                                handle, transA, transB, m, nrhs, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb, nullptr, ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO, nullptr),
                            "alphasparse_dcu_s_csrsm_solve");
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    std::cout << time << ",";

    hipMemcpy(matB_ict, dmatB, sizeof(float) * B_cols * ldb, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dmatB);

    alphasparse_dcu_destroy_mat_descr(descrA);
    alphasparse_dcu_destory_handle(handle);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    file   = args_get_data_file(argc, argv);
    check  = args_get_if_check(argc, argv);
    transA = alpha_args_get_transA(argc, argv);
    transB = alpha_args_get_transB(argc, argv);
    descr  = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_index_base_t csr_index;

    alphasparse_matrix_t coo, csr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    float *coo_values;
    // read coo
    alpha_read_coo(file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
    if (A_rows != A_cols) {
        printf("m != n\n");
        return 0;
    }

    columns = args_get_columns(argc, argv, A_cols); // 默认C是方阵

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, A_rows, A_cols, rnnz, coo_row_index, coo_col_index, coo_values),
        "alphasparse_s_create_coo");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(
        alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
        "alphasparse_convert_csr");
    // 获取csr格式里的数据
    alpha_call_exit(
        alphasparse_s_export_csr(csr, &csr_index, &A_rows, &A_cols, &csr_row_ptr, &csr_row_ptr_end, &csr_col_index, &csr_values),
        "alphasparse_s_export_csr");

    // if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    { // A是方阵
        if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            B_rows = A_rows;
            B_cols = columns;
            ldb    = A_rows;
        } else // transB, conjB, B转置就用方阵测
        {
            B_rows = columns;
            B_cols = A_rows;
            ldb    = columns;
        }
    }

    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        roctransA = rocsparse_operation_none;
    else if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        roctransA = rocsparse_operation_transpose;
    else
        roctransA = rocsparse_operation_conjugate_transpose;

    if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        roctransB = rocsparse_operation_none;
    else if (transB == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        roctransB = rocsparse_operation_transpose;
    else
        roctransB = rocsparse_operation_conjugate_transpose;

    // init B C
    matB_ict = (float *)alpha_malloc(B_cols * ldb * sizeof(float));
    matB_roc = (float *)alpha_malloc(B_cols * ldb * sizeof(float));

    alpha_fill_random_s(matB_ict, 1, B_cols * ldb);
    alpha_fill_random_s(matB_roc, 1, B_cols * ldb);

    alpha_trsm_dcu();

    if (check) {
        roc_trsm_dcu();
        // std::cout << B_cols * ldb << std::endl;
        // for (int i = 0; i < 100; i++)
        // {
        //     cout << "rocC:" << matB_roc[i] << " ictC:" << matB_ict[i] << endl;
        // }
        check_s((float *)matB_roc, B_cols * ldb, (float *)matB_ict, B_cols * ldb);
    }
    printf("\n");

    alpha_free(matB_ict);
    alpha_free(matB_roc);
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
