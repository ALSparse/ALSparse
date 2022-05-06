/**
 * @brief ict dcu mv csr test
 * @author HPCRC, ICT
 */

#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "rocsparse.h"

// #include "rocsparse_init.hpp"
// #include "rocsparse_random.hpp"
// #include "utility.hpp"

#include <hip/hip_runtime_api.h>

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <alphasparse_dcu.h>
#include "alphasparse/util/io.h"
#include "alphasparse/util/args.h"
#include "alphasparse/util/check.h"

const char *fileA, *fileB;
bool check;
int iter;

struct alpha_matrix_descr descrA;
struct alpha_matrix_descr descrB;
struct alpha_matrix_descr descrC;

// csr format
ALPHA_INT rm, rn, rnnzA, rnnzB;
ALPHA_INT *csr_row_ptr_A, *csr_row_ptr_end_A, *csr_col_index_A;
ALPHA_INT *csr_row_ptr_B, *csr_row_ptr_end_B, *csr_col_index_B;
ALPHA_INT *alpha_csr_row_ptr_C, *alpha_csr_col_index_C;
rocsparse_int *roc_csr_row_ptr_C, *roc_csr_col_index_C;
float *csr_values_A, *csr_values_B;
float *roc_csr_values_C, *alpha_csr_values_C;

ALPHA_INT roc_res_nnz, alpha_res_nnz;

// parms for kernel
const float alpha = 2.f;
const float beta  = 3.f;

ALPHA_INT lo, diag, hi;
ALPHA_INT64 ops;

const ALPHA_INT warm_up = 0;
const ALPHA_INT trials  = 1;
const int batch_size    = 1;

static void alpha_mv_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    rocsparse_int m    = rm;
    rocsparse_int n    = rn;
    rocsparse_int nnzA = rnnzA;
    rocsparse_int nnzB = rnnzB;

    // Generate problem
    std::vector<rocsparse_int> hAptr(m + 1);
    std::vector<rocsparse_int> hAcol(nnzA);
    std::vector<float> hAval(nnzA);
    std::vector<rocsparse_int> hBptr(m + 1);
    std::vector<rocsparse_int> hBcol(nnzB);
    std::vector<float> hBval(nnzB);

    for (int i = 0; i < m; i++) {
        hAptr[i] = csr_row_ptr_A[i];
        hBptr[i] = csr_row_ptr_B[i];
    }
    hAptr[m] = csr_row_ptr_end_A[m - 1];
    hBptr[m] = csr_row_ptr_end_B[m - 1];

    for (int i = 0; i < nnzA; i++) {
        hAcol[i] = csr_col_index_A[i];
        hAval[i] = csr_values_A[i];
    }
    for (int i = 0; i < nnzB; i++) {
        hBcol[i] = csr_col_index_B[i];
        hBval[i] = csr_values_B[i];
    }

    // Offload data to device
    rocsparse_int *dAptr = NULL, *dBptr = NULL, *dCptr = NULL;
    rocsparse_int *dAcol = NULL, *dBcol = NULL, *dCcol = NULL;
    float *dAval = NULL, *dBval = NULL, *dCval = NULL;

    hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void **)&dBptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnzA);
    hipMalloc((void **)&dBcol, sizeof(rocsparse_int) * nnzB);
    hipMalloc((void **)&dAval, sizeof(float) * nnzA);
    hipMalloc((void **)&dBval, sizeof(float) * nnzB);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnzA, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(float) * nnzA, hipMemcpyHostToDevice);
    hipMemcpy(dBptr, hBptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dBcol, hBcol.data(), sizeof(rocsparse_int) * nnzB, hipMemcpyHostToDevice);
    hipMemcpy(dBval, hBval.data(), sizeof(float) * nnzB, hipMemcpyHostToDevice);

    float halpha = alpha;
    float hbeta  = beta;

    // Matrix descriptor
    // todo：support general only
    rocsparse_mat_descr descr_A;
    rocsparse_mat_descr descr_B;
    rocsparse_mat_descr descr_C;

    rocsparse_create_mat_descr(&descr_A);
    rocsparse_create_mat_descr(&descr_B);
    rocsparse_create_mat_descr(&descr_C);

    // Obtain number of total non-zero entries in C and row pointers of C
    rocsparse_int nnzC;

    hipMalloc((void **)&dCptr, sizeof(rocsparse_int) * (m + 1));

    roc_call_exit(
        rocsparse_csrgeam_nnz(handle, m, n, descr_A, nnzA, dAptr, dAcol, descr_B, nnzB, dBptr, dBcol, descr_C, dCptr, &nnzC),
        "rocsparse_csrgeam_nnz");

    // Compute column indices and values of C
    hipMalloc((void **)&dCcol, sizeof(rocsparse_int) * nnzC);
    hipMalloc((void **)&dCval, sizeof(float) * nnzC);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrmv
        roc_call_exit(
            rocsparse_scsrgeam(handle, m, n, &alpha, descr_A, nnzA, dAval, dAptr, dAcol, &beta, descr_B, nnzB, dBval, dBptr, dBcol, descr_C, dCval, dCptr, dCcol),
            "rocsparse_scsrgeam");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    float time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call rocsparse csrmv
            roc_call_exit(
                rocsparse_scsrgeam(handle, m, n, &alpha, descr_A, nnzA, dAval, dAptr, dAcol, &beta, descr_B, nnzB, dBval, dBptr, dBcol, descr_C, dCval, dCptr, dCcol),
                "rocsparse_scsrgeam");
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    roc_csr_values_C = (float *)alpha_malloc(nnzC * sizeof(float));
    roc_csr_row_ptr_C =
        (rocsparse_int *)alpha_malloc((m + 1) * sizeof(rocsparse_int));
    roc_csr_col_index_C =
        (rocsparse_int *)alpha_malloc(nnzC * sizeof(rocsparse_int));

    hipMemcpy(roc_csr_values_C, dCval, sizeof(float) * nnzC, hipMemcpyDeviceToHost);
    hipMemcpy(roc_csr_row_ptr_C, dCptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyDeviceToHost);
    hipMemcpy(roc_csr_col_index_C, dCcol, sizeof(rocsparse_int) * nnzC, hipMemcpyDeviceToHost);
    roc_res_nnz = nnzC;

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);

    hipFree(dBptr);
    hipFree(dBcol);
    hipFree(dBval);

    hipFree(dCptr);
    hipFree(dCcol);
    hipFree(dCval);

    rocsparse_destroy_mat_descr(descr_A);
    rocsparse_destroy_mat_descr(descr_B);
    rocsparse_destroy_mat_descr(descr_C);
    rocsparse_destroy_handle(handle);
}

static void alpha_mv()
{
    // rocSPARSE handle
    alphasparse_dcu_handle_t handle;
    init_handle(&handle);
    alphasparse_dcu_get_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem
    ALPHA_INT m    = rm;
    ALPHA_INT n    = rn;
    ALPHA_INT nnzA = rnnzA, nnzB = rnnzB;

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
    ALPHA_INT *hBptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));

    for (int i = 0; i < m; i++) {
        hAptr[i] = csr_row_ptr_A[i];
        hBptr[i] = csr_row_ptr_B[i];
    }
    hAptr[m] = csr_row_ptr_end_A[m - 1];
    hBptr[m] = csr_row_ptr_end_B[m - 1];

    // Offload data to device
    ALPHA_INT *dAptr = NULL;
    ALPHA_INT *dAcol = NULL;
    float *dAval     = NULL;

    ALPHA_INT *dBptr = NULL;
    ALPHA_INT *dBcol = NULL;
    float *dBval     = NULL;

    ALPHA_INT *dCptr = NULL;
    ALPHA_INT *dCcol = NULL;
    float *dCval     = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnzA));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnzA));

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBcol, sizeof(ALPHA_INT) * nnzB));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBval, sizeof(float) * nnzB));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAcol, csr_col_index_A, sizeof(ALPHA_INT) * nnzA, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAval, csr_values_A, sizeof(float) * nnzA, hipMemcpyHostToDevice));

    PRINT_IF_HIP_ERROR(hipMemcpy(dBptr, hBptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dBcol, csr_col_index_B, sizeof(ALPHA_INT) * nnzB, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dBval, csr_values_B, sizeof(float) * nnzB, hipMemcpyHostToDevice));

    float halpha = alpha;
    float hbeta  = beta;

    // Matrix descriptor
    // todo: set general
    alpha_dcu_matrix_descr_t descr_A;
    alphasparse_dcu_create_mat_descr(&descr_A);
    alpha_dcu_matrix_descr_t descr_B;
    alphasparse_dcu_create_mat_descr(&descr_B);
    alpha_dcu_matrix_descr_t descr_C;
    alphasparse_dcu_create_mat_descr(&descr_C);

    // Obtain number of total non-zero entries in C and row pointers of C
    ALPHA_INT nnzC;

    hipMalloc((void **)&dCptr, sizeof(ALPHA_INT) * (m + 1));

    alpha_call_exit(alphasparse_dcu_csrgeam_nnz(handle, m, n, descr_A, nnzA, dAptr, dAcol, descr_B, nnzB, dBptr, dBcol, descr_C, dCptr, &nnzC),
                    "alphasparse_dcu_csrgeam_nnz");

    // Compute column indices and values of C
    hipMalloc((void **)&dCcol, sizeof(ALPHA_INT) * nnzC);
    hipMalloc((void **)&dCval, sizeof(float) * nnzC);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrmv
        alpha_call_exit(
            alphasparse_dcu_s_csrgeam(handle, m, n, &alpha, descr_A, nnzA, dAval, dAptr, dAcol, &beta, descr_B, nnzB, dBval, dBptr, dBcol, descr_C, dCval, dCptr, dCcol),
            "alphasparse_dcu_s_csrgeam");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    float time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call rocsparse csrmv
            roc_call_exit(
                alphasparse_dcu_s_csrgeam(handle, m, n, &alpha, descr_A, nnzA, dAval, dAptr, dAcol, &beta, descr_B, nnzB, dBval, dBptr, dBcol, descr_C, dCval, dCptr, dCcol),
                "alphasparse_dcu_s_csrgeam");
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    //            time      = (get_time_us() - time) / (trials * batch_size *
    //            1e3);
    //     double bandwidth = static_cast<double>(sizeof(double) * (2 * m + nnz) +
    //     sizeof(ALPHA_INT) * (m + 1 + nnz)) / time / 1e6; double gflops    =
    //     static_cast<double>(2 * nnz) / time / 1e6;

    //    std                                                       : :
    //    cout.precision(2); std : : cout.setf(std::ios::fixed); std : :
    //    cout.setf(std::ios::left); std : : cout << std                : : endl
    // << "### alphasparse_dcu_s_csrmv WITHOUT meta data ###" << std:: endl;
    //    std                                                       : : cout <<
    //    std                :                           : setw(12) << "m" << std
    //    : : setw(12) << "n" << std:: setw(12) << "nnz"
    // << std                                                       : : setw(12)
    // << "alpha" << std:: setw(12) << "beta" << std:: setw(12) << "GFlop/s"
    // << std                                                       : : setw(12)
    // << "GB/s" << std  :                           : setw(12) << "msec" << std::
    // endl;
    //    std                                                       : : cout <<
    //    std                :                           : setw(12) << m << std :
    //    : setw(12) << n << std   : : setw(12) << nnz << std:: setw(12)
    // << halpha << std                                             : : setw(12)
    // << hbeta << std   :                           : setw(12) << gflops << std::
    // setw(12)
    // << bandwidth << std                                          : : setw(12)
    // << time << std    :                           : endl;
    alpha_csr_values_C    = (float *)alpha_malloc(nnzC * sizeof(float));
    alpha_csr_row_ptr_C   = (ALPHA_INT *)alpha_malloc((m + 1) * sizeof(ALPHA_INT));
    alpha_csr_col_index_C = (ALPHA_INT *)alpha_malloc(nnzC * sizeof(ALPHA_INT));

    hipMemcpy(alpha_csr_values_C, dCval, sizeof(float) * nnzC, hipMemcpyDeviceToHost);
    hipMemcpy(alpha_csr_row_ptr_C, dCptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyDeviceToHost);
    hipMemcpy(alpha_csr_col_index_C, dCcol, sizeof(ALPHA_INT) * nnzC, hipMemcpyDeviceToHost);
    alpha_res_nnz = nnzC;

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);

    hipFree(dBptr);
    hipFree(dBcol);
    hipFree(dBval);

    hipFree(dCptr);
    hipFree(dCcol);
    hipFree(dCval);

    alphasparse_dcu_destroy_mat_descr(descr_A);
    alphasparse_dcu_destroy_mat_descr(descr_B);
    alphasparse_dcu_destroy_mat_descr(descr_C);
    alphasparse_dcu_destory_handle(handle);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    fileA  = args_get_data_fileA(argc, argv);
    fileB  = args_get_data_fileB(argc, argv);
    check  = args_get_if_check(argc, argv);
    iter   = args_get_iter(argc, argv);
    descrA = alpha_args_get_matrix_descrA(argc, argv);
    descrB = alpha_args_get_matrix_descrB(argc, argv);

    alphasparse_index_base_t csr_index;

    alphasparse_matrix_t coo, csrA, csrB;
    ALPHA_INT *coo_row_index, *coo_col_index;
    float *coo_values;
    // read coo
    alpha_read_coo(fileA, &rm, &rn, &rnnzA, &coo_row_index, &coo_col_index, &coo_values);
    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, rm, rn, rnnzA, coo_row_index, coo_col_index, coo_values),
        "alphasparse_s_create_cooA");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(
        alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
        "alphasparse_convert_csrA");
    // 获取csr格式里的数据
    alpha_call_exit(alphasparse_s_export_csr(csrA, &csr_index, &rm, &rn, &csr_row_ptr_A, &csr_row_ptr_end_A, &csr_col_index_A, &csr_values_A),
                    "alphasparse_s_export_csrA");
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);

    // read coo
    alpha_read_coo(fileB, &rm, &rn, &rnnzB, &coo_row_index, &coo_col_index, &coo_values);
    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, rm, rn, rnnzB, coo_row_index, coo_col_index, coo_values),
        "alphasparse_s_create_cooB");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(
        alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrB),
        "alphasparse_convert_csrB");
    // 获取csr格式里的数据
    alpha_call_exit(alphasparse_s_export_csr(csrB, &csr_index, &rm, &rn, &csr_row_ptr_B, &csr_row_ptr_end_B, &csr_col_index_B, &csr_values_B),
                    "alphasparse_s_export_csrB");
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);

    alpha_mv();

    if (check) {
        alpha_mv_dcu();
        cout << "\ncheck csr row ptr:\n"
             << endl;
        for (int i = 0; i < rm + 1; i++) {
            // checked
            // cout << "roc ptr: %d: " << roc_csr_row_ptr_C[i] << " ict ptr: %d:" <<
            // alpha_csr_row_ptr_C[i] << endl;
        }
        //check_ALPHA_INT_vec(roc_csr_row_ptr_C, rm + 1, alpha_csr_row_ptr_C, rm + 1);

        cout << "\ncheck csr col idx:\n"
             << endl;
        //check_ALPHA_INT_vec(roc_csr_col_index_C, roc_res_nnz, alpha_csr_col_index_C, alpha_res_nnz);

        cout << "\ncheck csr val:\n"
             << endl;
        check_s((float *)roc_csr_values_C, roc_res_nnz, (float *)alpha_csr_values_C, alpha_res_nnz);
    }
    printf("\n");

    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
