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
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <alpha_spblas_dcu.h>

const char *file;
bool check;

alphasparse_operation_t transA, transB;
rocsparse_operation roctransA, roctransB;
struct alpha_matrix_descr descr;
alphasparse_layout_t layout;

ALPHA_INT columns;
// csr format
ALPHA_INT A_rows, A_cols, rnnz;
ALPHA_INT *csr_row_ptr, *csr_row_ptr_end, *csr_col_index;
ALPHA_Complex8 *csr_values;

// parms for kernel
ALPHA_Complex8 *matB, *matC_ict, *matC_roc;
ALPHA_INT C_rows, C_cols, C_k;
ALPHA_INT B_cols;
ALPHA_INT ldb, ldc;
const ALPHA_Complex8 alpha = {2.f, 2.f};
const ALPHA_Complex8 beta  = {3.f, 3.f};

const ALPHA_INT warm_up = 5;
const ALPHA_INT trials  = 10;
const int batch_size    = 1;

static void roc_mm_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    // std::cout << "Device: " << devProp.name << std::endl;

    rocsparse_int m   = C_rows;
    rocsparse_int n   = C_cols;
    rocsparse_int k   = C_k;
    rocsparse_int nnz = rnnz;

    // Generate problem
    std::vector<rocsparse_int> hAptr(m + 1);
    std::vector<rocsparse_int> hAcol(nnz);
    std::vector<ALPHA_Complex8> hAval(nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    rocsparse_int *dAptr           = NULL;
    rocsparse_int *dAcol           = NULL;
    rocsparse_float_complex *dAval = NULL;
    rocsparse_float_complex *dmatB = NULL;
    rocsparse_float_complex *dmatC = NULL;

    hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void **)&dAval, sizeof(rocsparse_float_complex) * nnz);
    hipMalloc((void **)&dmatB, sizeof(rocsparse_float_complex) * B_cols * ldb);
    hipMalloc((void **)&dmatC, sizeof(rocsparse_float_complex) * C_cols * ldc);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(rocsparse_float_complex) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dmatB, matB, sizeof(rocsparse_float_complex) * B_cols * ldb, hipMemcpyHostToDevice);
    hipMemcpy(dmatC, matC_roc, sizeof(rocsparse_float_complex) * C_cols * ldc, hipMemcpyHostToDevice);

    rocsparse_float_complex halpha, hbeta;
    halpha.x = alpha.real;
    halpha.y = alpha.imag;
    hbeta.x  = beta.real;
    hbeta.y  = beta.imag;

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrmm
        rocsparse_ccsrmm(handle, roctransA, roctransB, m, n, k, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta, dmatC, ldc);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // cout << "m:" << m << " n:" << n << " k:" << k << " nnz:" << nnz << endl;
    // cout << "ldb:" << ldb << " ldc" << ldc << endl;
    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call rocsparse csrmm
            rocsparse_status x = rocsparse_ccsrmm(
                handle, roctransA, roctransB, m, n, k, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta, dmatC, ldc);
            if (x) {
                cout << "status num: \n"
                     << x << endl;
                exit(-1);
            }
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    std::cout << time << std::endl;

    hipMemcpy(matC_roc, dmatC, sizeof(ALPHA_Complex8) * C_cols * ldc, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dmatB);
    hipFree(dmatC);

    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);
}

static void alpha_mm_dcu()
{
    // rocSPARSE handle
    alphasparse_dcu_handle_t handle;
    init_handle(&handle);
    alphasparse_dcu_get_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    // std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem
    ALPHA_INT m   = C_rows;
    ALPHA_INT n   = C_cols;
    ALPHA_INT k   = C_k;
    ALPHA_INT nnz = rnnz;

    ALPHA_INT *hAptr      = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
    ALPHA_INT *hAcol      = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz);
    ALPHA_Complex8 *hAval = (ALPHA_Complex8 *)alpha_malloc(sizeof(ALPHA_Complex8) * nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    ALPHA_INT *dAptr      = NULL;
    ALPHA_INT *dAcol      = NULL;
    ALPHA_Complex8 *dAval = NULL;
    ALPHA_Complex8 *dmatB = NULL;
    ALPHA_Complex8 *dmatC = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(ALPHA_Complex8) * nnz));
    PRINT_IF_HIP_ERROR(
        hipMalloc((void **)&dmatB, sizeof(ALPHA_Complex8) * B_cols * ldb));
    PRINT_IF_HIP_ERROR(
        hipMalloc((void **)&dmatC, sizeof(ALPHA_Complex8) * C_cols * ldc));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAval, hAval, sizeof(ALPHA_Complex8) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dmatB, matB, sizeof(ALPHA_Complex8) * B_cols * ldb, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dmatC, matC_ict, sizeof(ALPHA_Complex8) * C_cols * ldc, hipMemcpyHostToDevice));

    ALPHA_Complex8 halpha = alpha;
    ALPHA_Complex8 hbeta  = beta;

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrA;
    alphasparse_dcu_create_mat_descr(&descrA);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu csrmm
        alphasparse_dcu_c_csrmm(handle, transA, transB, layout, m, n, k, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta, dmatC, ldc);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call alphasparse_dcu csrmm
            alphasparse_dcu_c_csrmm(handle, transA, transB, layout, m, n, k, nnz, &halpha, descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta, dmatC, ldc);
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    std::cout << time << ",";

    hipMemcpy(matC_ict, dmatC, sizeof(ALPHA_Complex8) * C_cols * ldc, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dmatB);
    hipFree(dmatC);

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
    layout = alpha_args_get_layout(argc, argv);

    alphasparse_index_base_t csr_index;

    alphasparse_matrix_t coo, csr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    ALPHA_Complex8 *coo_values;
    // read coo
    alpha_read_coo_c(file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
    columns = args_get_columns(argc, argv, A_cols); // 默认C是方阵

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, A_rows, A_cols, rnnz, coo_row_index, coo_col_index, coo_values),
        "alphasparse_c_create_coo");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(
        alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
        "alphasparse_convert_csr");
    // 获取csr格式里的数据
    alpha_call_exit(
        alphasparse_c_export_csr(csr, &csr_index, &A_rows, &A_cols, &csr_row_ptr, &csr_row_ptr_end, &csr_col_index, &csr_values),
        "alphasparse_c_export_csr");

    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
        if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            C_rows = A_rows;
            C_cols = columns;
            C_k    = A_cols;
            B_cols = columns;
            ldb    = A_cols;
            ldc    = A_rows;
        } else {
            C_rows = A_rows;
            C_cols = columns;
            C_k    = A_cols;
            B_cols = A_cols;
            ldb    = columns;
            ldc    = A_rows;
        }
    } else // transA, conjA
    {
        if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            C_rows = A_cols;
            C_cols = columns;
            C_k    = A_rows;
            B_cols = columns;
            ldb    = A_rows;
            ldc    = C_rows;
        } else // transB, conjB, B转置就用方阵测
        {
            C_rows = A_rows;
            C_cols = columns;
            C_k    = A_rows;
            B_cols = A_rows;
            ldb    = columns;
            ldc    = C_rows;
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
    matB     = (ALPHA_Complex8 *)alpha_malloc(B_cols * ldb * sizeof(ALPHA_Complex8));
    matC_ict = (ALPHA_Complex8 *)alpha_malloc(C_cols * ldc * sizeof(ALPHA_Complex8));
    matC_roc = (ALPHA_Complex8 *)alpha_malloc(C_cols * ldc * sizeof(ALPHA_Complex8));

    alpha_fill_random_c(matB, 0, B_cols * ldb);
    alpha_fill_random_c(matC_ict, 1, C_cols * ldc);
    alpha_fill_random_c(matC_roc, 1, C_cols * ldc);

    alpha_mm_dcu();

    if (check) {
        roc_mm_dcu();
        // for (int i = 0; i < 100; i++)
        // {
        //     cout << "rocC:" << matC_roc[i] << " ictC:" << matC_ict[i] << endl;
        // }
        check_c((ALPHA_Complex8 *)matC_roc, C_cols * ldc, (ALPHA_Complex8 *)matC_ict, C_cols * ldc);
    }
    printf("\n");

    alpha_free(matB);
    alpha_free(matC_ict);
    alpha_free(matC_roc);
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
