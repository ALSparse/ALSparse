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

ALPHA_INT A_rows;
// csr format
ALPHA_INT B_rows, B_cols, rnnz;
ALPHA_INT *csr_row_ptr, *csr_row_ptr_end, *csr_col_index;
ALPHA_Complex16 *csr_values;

// parms for kernel
ALPHA_Complex16 *matA, *matC_ict, *matC_roc;
ALPHA_INT C_rows, C_cols, C_k;
ALPHA_INT lda, ldc;
const ALPHA_Complex16 alpha = {2.f, 2.f};
const ALPHA_Complex16 beta  = {3.f, 3.f};

const ALPHA_INT warm_up = 5;
const ALPHA_INT trials  = 10;

static void roc_mmi_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);

    rocsparse_int m   = C_rows;
    rocsparse_int n   = C_cols;
    rocsparse_int k   = C_k;
    rocsparse_int nnz = rnnz;

    // Generate problem
    std::vector<rocsparse_int> hAptr(C_k + 1);
    std::vector<rocsparse_int> hAcol(nnz);
    std::vector<ALPHA_Complex16> hAval(nnz);

    for (int i = 0; i < C_k; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[C_k] = csr_row_ptr_end[C_k - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    rocsparse_int *dAptr            = NULL;
    rocsparse_int *dAcol            = NULL;
    rocsparse_double_complex *dAval = NULL;
    rocsparse_double_complex *dmatA = NULL;
    rocsparse_double_complex *dmatC = NULL;

    hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (C_k + 1));
    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void **)&dAval, sizeof(rocsparse_double_complex) * nnz);
    hipMalloc((void **)&dmatA, sizeof(rocsparse_double_complex) * C_k * lda);
    hipMalloc((void **)&dmatC, sizeof(rocsparse_double_complex) * C_cols * ldc);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (C_k + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(rocsparse_double_complex) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dmatA, matA, sizeof(rocsparse_double_complex) * C_k * lda, hipMemcpyHostToDevice);
    hipMemcpy(dmatC, matC_roc, sizeof(rocsparse_double_complex) * C_cols * ldc, hipMemcpyHostToDevice);

    rocsparse_double_complex halpha, hbeta;
    halpha.x = alpha.real;
    halpha.y = alpha.imag;
    hbeta.x  = beta.real;
    hbeta.y  = beta.imag;

    // Matrix descriptor
    rocsparse_mat_descr descrB;
    rocsparse_create_mat_descr(&descrB);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrmmi
        roc_call_exit(rocsparse_zgemmi(handle, roctransA, roctransB, m, n, k, nnz, &halpha, dmatA, lda, descrB, dAval, dAptr, dAcol, &hbeta, dmatC, ldc),
                      "rocsparse_zgemmi");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // cout << "m:" << m << " n:" << n << " k:" << k << " nnz:" << nnz << endl;
    // cout << "lda:" << lda << " ldc" << ldc << endl;
    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        // Call rocsparse csrmmi
        roc_call_exit(rocsparse_zgemmi(handle, roctransA, roctransB, m, n, k, nnz, &halpha, dmatA, lda, descrB, dAval, dAptr, dAcol, &hbeta, dmatC, ldc),
                      "rocsparse_zgemmi");

        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * 1e3);
    std::cout << time << std::endl;

    hipMemcpy(matC_roc, dmatC, sizeof(ALPHA_Complex16) * C_cols * ldc, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dmatA);
    hipFree(dmatC);

    rocsparse_destroy_mat_descr(descrB);
    rocsparse_destroy_handle(handle);
}

static void alpha_mmi_dcu()
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

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (C_k + 1));
    ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz);
    ALPHA_Complex16 *hAval =
        (ALPHA_Complex16 *)alpha_malloc(sizeof(ALPHA_Complex16) * nnz);

    for (int i = 0; i < C_k; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[C_k] = csr_row_ptr_end[C_k - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    ALPHA_INT *dAptr       = NULL;
    ALPHA_INT *dAcol       = NULL;
    ALPHA_Complex16 *dAval = NULL;
    ALPHA_Complex16 *dmatA = NULL;
    ALPHA_Complex16 *dmatC = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (C_k + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(ALPHA_Complex16) * nnz));
    PRINT_IF_HIP_ERROR(
        hipMalloc((void **)&dmatA, sizeof(ALPHA_Complex16) * C_k * lda));
    PRINT_IF_HIP_ERROR(
        hipMalloc((void **)&dmatC, sizeof(ALPHA_Complex16) * C_cols * ldc));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (C_k + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAval, hAval, sizeof(ALPHA_Complex16) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dmatA, matA, sizeof(ALPHA_Complex16) * C_k * lda, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dmatC, matC_ict, sizeof(ALPHA_Complex16) * C_cols * ldc, hipMemcpyHostToDevice));

    ALPHA_Complex16 halpha = alpha;
    ALPHA_Complex16 hbeta  = beta;

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrB;
    alphasparse_dcu_create_mat_descr(&descrB);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu csrmmi
        alpha_call_exit(alphasparse_dcu_z_gemmi(handle, transA, transB, m, n, k, nnz, &halpha, dmatA, lda, descrB, dAval, dAptr, dAcol, &hbeta, dmatC, ldc),
                        "alphasparse_dcu_z_gemmi");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // cout << "m:" << m << " n:" << n << " k:" << k << " nnz:" << nnz << endl;
    // cout << "lda:" << lda << " ldc" << ldc << endl;
    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        // Call alphasparse_dcu csrmmi
        alpha_call_exit(alphasparse_dcu_z_gemmi(handle, transA, transB, m, n, k, nnz, &halpha, dmatA, lda, descrB, dAval, dAptr, dAcol, &hbeta, dmatC, ldc),
                        "alphasparse_dcu_z_gemmi");
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * 1e3);
    std::cout << time << ",";

    hipMemcpy(matC_ict, dmatC, sizeof(ALPHA_Complex16) * C_cols * ldc, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dmatA);
    hipFree(dmatC);

    alphasparse_dcu_destroy_mat_descr(descrB);
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

    alphasparse_index_base_t csr_index;

    alphasparse_matrix_t coo, csr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    ALPHA_Complex16 *coo_values;
    // read coo
    alpha_read_coo_z(file, &B_rows, &B_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
    A_rows = args_get_columns(argc, argv, B_rows); // 稠密矩阵A的行数

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, B_rows, B_cols, rnnz, coo_row_index, coo_col_index, coo_values),
        "alphasparse_z_create_coo");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(
        alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
        "alphasparse_convert_csr");
    // 获取csr格式里的数据
    alpha_call_exit(
        alphasparse_z_export_csr(csr, &csr_index, &B_rows, &B_cols, &csr_row_ptr, &csr_row_ptr_end, &csr_col_index, &csr_values),
        "alphasparse_z_export_csr");

    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
        if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            C_rows = A_rows;
            C_cols = B_cols;
            C_k    = B_rows;
            lda    = A_rows;
            ldc    = A_rows;
        } else // transB, conjB
        {
            C_rows = A_rows;
            C_cols = B_rows;
            C_k    = B_cols;
            lda    = A_rows;
            ldc    = A_rows;
        }
    } else // transA, conjA, A转置就用方阵测
    {
        if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            C_rows = B_rows;
            C_cols = B_cols;
            C_k    = B_rows;
            lda    = B_rows;
            ldc    = B_rows;
        } else // transB, conjB,
        {
            C_rows = B_cols;
            C_cols = B_rows;
            C_k    = B_cols;
            lda    = B_cols;
            ldc    = B_cols;
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
    matA     = (ALPHA_Complex16 *)alpha_malloc(C_k * lda * sizeof(ALPHA_Complex16));
    matC_ict = (ALPHA_Complex16 *)alpha_malloc(C_cols * ldc * sizeof(ALPHA_Complex16));
    matC_roc = (ALPHA_Complex16 *)alpha_malloc(C_cols * ldc * sizeof(ALPHA_Complex16));

    alpha_fill_random_z(matA, 0, C_k * lda);
    alpha_fill_random_z(matC_ict, 1, C_cols * ldc);
    alpha_fill_random_z(matC_roc, 1, C_cols * ldc);

    alpha_mmi_dcu();

    if (check) {
        roc_mmi_dcu();
        check_z((ALPHA_Complex16 *)matC_roc, C_cols * ldc, (ALPHA_Complex16 *)matC_ict, C_cols * ldc);
    }
    printf("\n");

    alpha_free(matA);
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
