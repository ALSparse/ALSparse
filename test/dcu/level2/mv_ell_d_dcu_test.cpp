/**
 * @brief ict dcu mv ell test
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

ALPHA_INT nnz;
// ell format
ALPHA_INT rm, rk, ell_width;
ALPHA_INT *ell_col_idx;
double *ell_values;

// parms for kernel
double *x;
double *icty;
double *rocy, *rocy_coo;
ALPHA_INT sizex, sizey;
const double alpha = 2.f;
const double beta  = 3.f;

const ALPHA_INT warm_up = 5;
const ALPHA_INT trials  = 10;
const int batch_size    = 1;

static void roc_mv_dcu()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    // std::cout << "Device: " << devProp.name << std::endl;

    rocsparse_int m        = rm;
    rocsparse_int n        = rk;
    rocsparse_int ell_wdth = ell_width;

    // Generate problem
    std::vector<rocsparse_int> hAcol(m * ell_wdth);
    for (int i = 0; i < m * ell_wdth; i++) {
        hAcol[i] = ell_col_idx[i];
    }

    // Offload data to device
    rocsparse_int *dAcol = NULL;
    double *dAval        = NULL;
    double *dx           = NULL;
    double *dy           = NULL;

    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * m * ell_wdth);
    hipMalloc((void **)&dAval, sizeof(double) * m * ell_wdth);
    hipMalloc((void **)&dx, sizeof(double) * sizex);
    hipMalloc((void **)&dy, sizeof(double) * sizey);

    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * m * ell_wdth, hipMemcpyHostToDevice);
    hipMemcpy(dAval, ell_values, sizeof(double) * m * ell_wdth, hipMemcpyHostToDevice);
    hipMemcpy(dx, x, sizeof(double) * sizex, hipMemcpyHostToDevice);
    hipMemcpy(dy, rocy, sizeof(double) * sizey, hipMemcpyHostToDevice);

    double halpha = alpha;
    double hbeta  = beta;

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse ellmv
        roc_call_exit(
            rocsparse_dellmv(handle, rocsparse_operation_none, m, n, &halpha, descrA, dAval, dAcol, ell_wdth, dx, &hbeta, dy),
            "rocsparse_sellmv");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // ell matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call rocsparse ellmv
            roc_call_exit(
                rocsparse_dellmv(handle, rocsparse_operation_none, m, n, &halpha, descrA, dAval, dAcol, ell_wdth, dx, &hbeta, dy),
                "rocsparse_sellmv");
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    std::cout << time << std::endl;

    hipMemcpy(rocy, dy, sizeof(double) * sizey, hipMemcpyDeviceToHost);

    // Clear up on device
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
    // std::cout << "Device: " << devProp.name << std::endl;

    // Generate problem
    ALPHA_INT m = rm;
    ALPHA_INT n = rk;

    // Offload data to device
    ALPHA_INT *dAcol = NULL;
    double *dAval    = NULL;
    double *dx       = NULL;
    double *dy       = NULL;

    PRINT_IF_HIP_ERROR(
        hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * m * ell_width));
    PRINT_IF_HIP_ERROR(
        hipMalloc((void **)&dAval, sizeof(double) * m * ell_width));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dx, sizeof(double) * sizex));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dy, sizeof(double) * sizey));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAcol, ell_col_idx, sizeof(ALPHA_INT) * m * ell_width, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAval, ell_values, sizeof(double) * m * ell_width, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dx, x, sizeof(double) * sizex, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dy, icty, sizeof(double) * sizey, hipMemcpyHostToDevice));

    double halpha = alpha;
    double hbeta  = beta;

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrA;
    alphasparse_dcu_create_mat_descr(&descrA);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu ellmv
        alphasparse_dcu_d_ellmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, &halpha, descrA, dAval, dAcol, ell_width, dx, &hbeta, dy);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // ell matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call alphasparse_dcu ellmv
            alphasparse_dcu_d_ellmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, &halpha, descrA, dAval, dAcol, ell_width, dx, &hbeta, dy);
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    std::cout << time << ",";

    hipMemcpy(icty, dy, sizeof(double) * sizey, hipMemcpyDeviceToHost);

    // Clear up on device
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

    alphasparse_index_base_t ell_index;

    alphasparse_matrix_t coo, ell;
    ALPHA_INT *coo_row_index, *coo_col_idx;
    double *coo_values;
    // read coo
    alpha_read_coo_d(file, &rm, &rk, &nnz, &coo_row_index, &coo_col_idx, &coo_values);

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, rm, rk, nnz, coo_row_index, coo_col_idx, coo_values),
        "alphasparse_d_create_coo");
    // 将稀疏矩阵从coo格式转换成ell格式
    alpha_call_exit(
        alphasparse_convert_ell(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &ell),
        "alphasparse_convert_ell");
    alpha_call_exit(alphasparse_d_export_ell(ell, &ell_index, &rm, &rk, &ell_width, &ell_col_idx, &ell_values),
                    "alphasparse_d_export_ell");

    sizex = rk, sizey = rm;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
        transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        sizex = rm;
        sizey = rk;
    }
    // init x y
    x        = (double *)alpha_malloc(sizex * sizeof(double));
    icty     = (double *)alpha_malloc(sizey * sizeof(double));
    rocy     = (double *)alpha_malloc(sizey * sizeof(double));
    rocy_coo = (double *)alpha_malloc(sizey * sizeof(double));

    alpha_fill_random_d(x, 4, sizex);
    alpha_fill_random_d(icty, 1, sizey);
    alpha_fill_random_d(rocy, 1, sizey);
    alpha_fill_random_d(rocy_coo, 1, sizey);

    alpha_mv_dcu();

    if (check) {
        roc_mv_dcu();
        // alpha_mv_roc_coo(coo_row_index, coo_col_idx, coo_values); //结果与coo一致

        // for (int i = 0; i < sizey; i++)
        // {
        //     std::cout << "rocy: " << rocy[i] << " icty: " << icty[i] <<
        //     std::endl;
        // }
        // roc_mv_coo_dcu(rm, rk, nnz, coo_row_index, coo_col_idx, coo_values);
        // std::cout << "check with rocsparse ell:";
        check_d((double *)rocy, sizey, (double *)icty, sizey);
        // std::cout << "check with rocsparse coo:";
        // check_d((double *)rocy_coo, sizey, (double *)icty, sizey);
    }
    printf("\n");

    alpha_free(x);
    alpha_free(icty);
    alpha_free(rocy);
    alpha_free(coo_row_index);
    alpha_free(coo_col_idx);
    alpha_free(coo_values);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
