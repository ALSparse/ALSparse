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

using namespace std;

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <alphasparse_dcu.h>

const char *file;
int thread_num;
bool check;
int iter;

alphasparse_operation_t transA;
alphasparse_layout_t layout;
struct alpha_matrix_descr descr;
rocsparse_operation roctransA = rocsparse_operation_none;

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

const ALPHA_INT warm_up = 5;
const ALPHA_INT trials  = 10;
const int batch_size    = 1;

static void roc_trsv_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    // std::cout << "Device: " << devProp.name << std::endl;

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

    // Obtain required buffer size
    size_t buffer_size;
    void *temp_buffer;

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

    rocsparse_double_complex halpha;
    halpha.x = alpha.real;
    halpha.y = alpha.imag;

    double time1 = get_time_us();
    // Obtain required buffer size
    rocsparse_mat_info info;
    rocsparse_create_mat_info(&info);
    rocsparse_zcsrsv_buffer_size(handle, roctransA, m, nnz, descrA, dAval, dAptr, dAcol, info, &buffer_size);

    // Allocate temporary buffer
    hipMalloc((void **)&temp_buffer, buffer_size);

    // Perform analysis step
    roc_call_exit(
        rocsparse_zcsrsv_analysis(handle, roctransA, m, nnz, descrA, dAval, dAptr, dAcol, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, temp_buffer),
        "rocsparse_zcsrsv_analysis");
    hipDeviceSynchronize();
    time1 = (get_time_us() - time1) / (1 * 1 * 1e3);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse csrtrsv
        roc_call_exit(
            rocsparse_zcsrsv_solve(handle, roctransA, m, nnz, &halpha, descrA, dAval, dAptr, dAcol, info, dx, dy, rocsparse_solve_policy_auto, temp_buffer),
            "rocsparse_zcsrsv_solve");
    }

    // Device synchronization
    hipDeviceSynchronize();
    // Start time measurement
    double time2 = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call rocsparse csrtrsv
            roc_call_exit(
                rocsparse_zcsrsv_solve(handle, roctransA, m, nnz, &halpha, descrA, dAval, dAptr, dAcol, info, dx, dy, rocsparse_solve_policy_auto, temp_buffer),
                "rocsparse_zcsrsv_solve");
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time2 = (get_time_us() - time2) / (trials * batch_size * 1e3);
    std::cout << time1 + time2 << std::endl;

    hipMemcpy(rocy, dy, sizeof(ALPHA_Complex16) * sizey, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);
}

static void alpha_trsv_dcu()
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

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrA;
    alphasparse_dcu_create_mat_descr(&descrA);
    descrA->diag = descr.diag;
    descrA->mode = descr.mode;
    descrA->type = descr.type;

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu csrmv
        alpha_call_exit(
            alphasparse_dcu_z_csrsv_solve(handle, transA, m, nnz, &halpha, descrA, dAval, dAptr, dAcol, NULL, dx, dy, ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO, NULL),
            "alphasparse_dcu_z_csrsv_solve");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call alphasparse_dcu csrmv
            alpha_call_exit(
                alphasparse_dcu_z_csrsv_solve(handle, transA, m, nnz, &halpha, descrA, dAval, dAptr, dAcol, NULL, dx, dy, ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO, NULL),
                "alphasparse_dcu_z_csrsv_solve");
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);

    std::cout << time << ",";

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
    file       = args_get_data_file(argc, argv);
    thread_num = args_get_thread_num(argc, argv);
    check      = args_get_if_check(argc, argv);
    transA     = alpha_args_get_transA(argc, argv);
    descr      = alpha_args_get_matrix_descrA(argc, argv);

    alphasparse_index_base_t csr_index;

    alphasparse_matrix_t coo, csr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    ALPHA_Complex16 *coo_values;
    // read coo
    alpha_read_coo_z(file, &rm, &rk, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
    if (rm != rk) {
        printf("m != n\n");
        return 0;
    }

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
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        sizex     = rm;
        sizey     = rk;
        roctransA = rocsparse_operation_transpose;
    } else if (transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        sizex     = rm;
        sizey     = rk;
        roctransA = rocsparse_operation_conjugate_transpose;
    }

    // init x y
    x    = (ALPHA_Complex16 *)alpha_malloc(sizex * sizeof(ALPHA_Complex16));
    icty = (ALPHA_Complex16 *)alpha_malloc(sizey * sizeof(ALPHA_Complex16));
    rocy = (ALPHA_Complex16 *)alpha_malloc(sizey * sizeof(ALPHA_Complex16));

    alpha_fill_random_z(x, 0, sizex);
    alpha_fill_random_z(icty, 1, sizey);
    alpha_fill_random_z(rocy, 1, sizey);
    // memset(x, 0, sizex);
    // memset(icty, 0, sizey);
    // memset(rocy, 0, sizey);

    alpha_trsv_dcu();
    if (check) {
        roc_trsv_dcu();
        // int len = sizey < 100 ? sizey : 100;
        // for(int i = 0; i < len; i++)
        //{
        //     std::cout << rocy[i] << " " << icty[i] << std::endl;
        // }
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
