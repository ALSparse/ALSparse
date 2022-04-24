/**
 * @brief ict dcu mv bsr test
 * @author HPCRC, ICT
 */


#include <hip/hip_runtime_api.h>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <alphasparse_dcu.h>

const char *file;
bool check;

alphasparse_operation_t transA;
alphasparse_layout_t layout;
struct alpha_matrix_descr descr;

// csr format
ALPHA_INT rm, rk, rnnz;
ALPHA_INT *csr_row_ptr, *csr_row_ptr_end, *csr_col_index;
ALPHA_Complex16 *csr_values;

// parms for kernel
ALPHA_Complex16 *x;
ALPHA_Complex16 *icty;
ALPHA_Complex16 *rocy;
ALPHA_INT sizex, sizey;
const ALPHA_Complex16 alpha = {2.f, 2.f};
const ALPHA_Complex16 beta  = {3.f, 3.f};

const ALPHA_INT warm_up = 5;
ALPHA_INT trials        = 10;

static void roc_mv_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);

    rocsparse_int m   = rm;
    rocsparse_int n   = rk;
    rocsparse_int nnz = rnnz;

    // Generate problem
    std::vector<rocsparse_int> hAptr(m + 1);
    std::vector<rocsparse_int> hAcol(nnz);
    std::vector<ALPHA_Complex16> hAval(nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    rocsparse_int *dAptr            = NULL;
    rocsparse_int *dAcol            = NULL;
    rocsparse_double_complex *dAval = NULL;
    rocsparse_double_complex *dx    = NULL;
    rocsparse_double_complex *dy    = NULL;

    hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void **)&dAval, sizeof(rocsparse_double_complex) * nnz);
    hipMalloc((void **)&dx, sizeof(rocsparse_double_complex) * sizex);
    hipMalloc((void **)&dy, sizeof(rocsparse_double_complex) * sizey);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(rocsparse_double_complex) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx, x, sizeof(rocsparse_double_complex) * sizex, hipMemcpyHostToDevice);
    hipMemcpy(dy, rocy, sizeof(rocsparse_double_complex) * sizey, hipMemcpyHostToDevice);

    rocsparse_double_complex halpha, hbeta;
    halpha.x = alpha.real;
    halpha.y = alpha.imag;
    hbeta.y  = beta.real;
    hbeta.y  = beta.imag;

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrmv
        roc_call_exit(
            rocsparse_zcsrmv(handle, rocsparse_operation_none, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, nullptr, dx, &hbeta, dy),
            "rocsparse_zcsrmv");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        // Call rocsparse csrmv
        roc_call_exit(rocsparse_zcsrmv(handle, rocsparse_operation_none, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, nullptr, dx, &hbeta, dy),
                      "rocsparse_zcsrmv");

        // Device synchronization
        hipDeviceSynchronize();
    }

    time             = (get_time_us() - time) / (trials * 1e3);
    double bandwidth = static_cast<double>(sizeof(rocsparse_double_complex) * (2 * m + nnz) + sizeof(rocsparse_int) * (m + 1 + nnz)) / time / 1e6;
    double gflops    = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout << time;
    // std::cout << " " << bandwidth;
    // std::cout << " " << gflops;
    std::cout << std::endl;

    hipMemcpy(rocy, dy, sizeof(rocsparse_double_complex) * sizey, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);
}

static void alpha_mv_dcu()
{
    // rocSPARSE handle
    alphasparse_dcu_handle_t handle;
    init_handle(&handle);
    alphasparse_dcu_get_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);

    // Generate problem
    ALPHA_INT m   = rm;
    ALPHA_INT n   = rk;
    ALPHA_INT nnz = rnnz;

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
    ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz);
    ALPHA_Complex16 *hAval =
        (ALPHA_Complex16 *)alpha_malloc(sizeof(ALPHA_Complex16) * nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = csr_row_ptr[i];

    hAptr[m] = csr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = csr_col_index[i];
        hAval[i] = csr_values[i];
    }

    // Offload data to device
    ALPHA_INT *dAptr       = NULL;
    ALPHA_INT *dAcol       = NULL;
    ALPHA_Complex16 *dAval = NULL;
    ALPHA_Complex16 *dx    = NULL;
    ALPHA_Complex16 *dy    = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(ALPHA_Complex16) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dx, sizeof(ALPHA_Complex16) * sizex));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dy, sizeof(ALPHA_Complex16) * sizey));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAval, hAval, sizeof(ALPHA_Complex16) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dx, x, sizeof(ALPHA_Complex16) * sizex, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dy, icty, sizeof(ALPHA_Complex16) * sizey, hipMemcpyHostToDevice));

    ALPHA_Complex16 halpha = alpha;
    ALPHA_Complex16 hbeta  = beta;

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrA;
    alphasparse_dcu_create_mat_descr(&descrA);

    alphasparse_dcu_mat_info_t info          = (alphasparse_dcu_mat_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_mat_info));
    alphasparse_dcu_csrmv_info_t csrmv_info  = (alphasparse_dcu_csrmv_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_csrmv_info));
    info->csrmv_info                         = csrmv_info;
    info->csrmv_info->csr_adaptive_has_tuned = false;

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu csrmv
        alpha_call_exit(
            alphasparse_dcu_z_csrmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, info, dx, &hbeta, dy),
            "alphasparse_dcu_z_csrmv");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        // Call alphasparse_dcu csrmv
        alpha_call_exit(
            alphasparse_dcu_z_csrmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &halpha, descrA, dAval, dAptr, dAcol, info, dx, &hbeta, dy),
            "alphasparse_dcu_z_csrmv");
        // Device synchronization
        hipDeviceSynchronize();
    }

    time             = (get_time_us() - time) / (trials * 1e3);
    double bandwidth = static_cast<double>(sizeof(ALPHA_Complex16)) * (2 * m + nnz) + sizeof(ALPHA_INT) * (m + 1 + nnz) / time / 1e6;
    double gflops    = static_cast<double>(2 * nnz) / time / 1e6;

    std::cout << time;
    // std::cout << " " << bandwidth;
    // std::cout << " " << gflops;
    std::cout << ",";

    hipMemcpy(icty, dy, sizeof(ALPHA_Complex16) * sizey, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

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
    descr  = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_index_base_t csr_index;

    alphasparse_matrix_t coo, csr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    ALPHA_Complex16 *coo_values;
    // read coo
    alpha_read_coo_z(file, &rm, &rk, &rnnz, &coo_row_index, &coo_col_index, &coo_values);

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, rm, rk, rnnz, coo_row_index, coo_col_index, coo_values),
        "alphasparse_z_create_coo");
    // 将稀疏矩阵从coo格式转换成csr格式
    alpha_call_exit(
        alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
        "alphasparse_convert_csr");
    // 获取csr格式里的数据
    alpha_call_exit(
        alphasparse_z_export_csr(csr, &csr_index, &rm, &rk, &csr_row_ptr, &csr_row_ptr_end, &csr_col_index, &csr_values),
        "alphasparse_z_export_csr");

    sizex = rk, sizey = rm;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        sizex = rm;
        sizey = rk;
    }
    // init x y
    x    = (ALPHA_Complex16 *)alpha_malloc(sizex * sizeof(ALPHA_Complex16));
    icty = (ALPHA_Complex16 *)alpha_malloc(sizey * sizeof(ALPHA_Complex16));
    rocy = (ALPHA_Complex16 *)alpha_malloc(sizey * sizeof(ALPHA_Complex16));

    alpha_fill_random_z(x, 0, sizex);
    alpha_fill_random_z(icty, 1, sizey);
    alpha_fill_random_z(rocy, 1, sizey);

    alpha_mv_dcu();

    if (check) {
        roc_mv_dcu();

        check_z((ALPHA_Complex16 *)rocy, sizey, (ALPHA_Complex16 *)icty, sizey);
    }
    printf("\n");

    alpha_free(x);
    alpha_free(icty);
    alpha_free(rocy);
    // alpha_free(csr_row_ptr);
    // alpha_free(csr_row_ptr_end);
    // alpha_free(csr_col_index);
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
