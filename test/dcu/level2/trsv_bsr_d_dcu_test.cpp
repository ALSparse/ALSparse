/**
 * @brief ict dcu mv bsr test
 * @author HPCRC, ICT
 */

#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "rocsparse.h"
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
rocsparse_direction roclayout = rocsparse_direction_row;

// bsr format
ALPHA_INT mb, nb, nnzb, bsr_dim = 2;
ALPHA_INT *bsr_row_ptr, *bsr_row_ptr_end, *bsr_col_index;
double *bsr_values;

// parms for kernel
double *x;
double *icty;
double *rocy;
ALPHA_INT sizex, sizey;
const double alpha = 2.f;

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

    rocsparse_int m   = mb;
    rocsparse_int n   = nb;
    rocsparse_int nnz = nnzb;

    // Generate problem
    std::vector<rocsparse_int> hAptr(m + 1);
    std::vector<rocsparse_int> hAcol(nnz);
    std::vector<double> hAval(nnz * bsr_dim * bsr_dim);

    for (int i = 0; i < m; i++)
        hAptr[i] = bsr_row_ptr[i];

    hAptr[m] = bsr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAcol[i] = bsr_col_index[i];
    }
    for (int i = 0; i < nnz * bsr_dim * bsr_dim; i++) {
        hAval[i] = bsr_values[i];
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
    rocsparse_int *dAptr = NULL;
    rocsparse_int *dAcol = NULL;
    double *dAval        = NULL;
    double *dx           = NULL;
    double *dy           = NULL;

    hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
    hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnz);
    hipMalloc((void **)&dAval, sizeof(double) * nnz * bsr_dim * bsr_dim);
    hipMalloc((void **)&dx, sizeof(double) * sizex);
    hipMalloc((void **)&dy, sizeof(double) * sizey);

    hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz * bsr_dim * bsr_dim, hipMemcpyHostToDevice);
    hipMemcpy(dx, x, sizeof(double) * sizex, hipMemcpyHostToDevice);
    hipMemcpy(dy, rocy, sizeof(double) * sizey, hipMemcpyHostToDevice);

    double halpha = alpha;

    double time1 = get_time_us();
    // Obtain required buffer size
    rocsparse_mat_info info;
    rocsparse_create_mat_info(&info);
    roc_call_exit(rocsparse_dbsrsv_buffer_size(handle, roclayout, roctransA, mb, nnzb, descrA, dAval, dAptr, dAcol, bsr_dim, info, &buffer_size),
                  "rocsparse_dbsrsv_buffer_size");

    // Allocate temporary buffer
    hipMalloc((void **)&temp_buffer, buffer_size);

    // Perform analysis step
    roc_call_exit(
        rocsparse_dbsrsv_analysis(handle, roclayout, roctransA, mb, nnzb, descrA, dAval, dAptr, dAcol, bsr_dim, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, temp_buffer),
        "rocsparse_dbsrsv_analysis");
    hipDeviceSynchronize();
    time1 = (get_time_us() - time1) / (1e3);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call rocsparse bsrtrsv
        roc_call_exit(
            rocsparse_dbsrsv_solve(handle, roclayout, roctransA, mb, nnzb, &alpha, descrA, dAval, dAptr, dAcol, bsr_dim, info, dx, dy, rocsparse_solve_policy_auto, temp_buffer),
            "rocsparse_dbsrsv_solve");
    }

    // Device synchronization
    hipDeviceSynchronize();
    // Start time measurement
    double time2 = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call rocsparse bsrtrsv
            roc_call_exit(
                rocsparse_dbsrsv_solve(handle, roclayout, roctransA, mb, nnzb, &alpha, descrA, dAval, dAptr, dAcol, bsr_dim, info, dx, dy, rocsparse_solve_policy_auto, temp_buffer),
                "rocsparse_dbsrsv_solve");
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time2 = (get_time_us() - time2) / (trials * batch_size * 1e3);
    std::cout << time1 + time2 << std::endl;

    hipMemcpy(rocy, dy, sizeof(double) * sizey, hipMemcpyDeviceToHost);

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
    ALPHA_INT m   = mb;
    ALPHA_INT n   = nb;
    ALPHA_INT nnz = nnzb;

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));

    for (int i = 0; i < m; i++)
        hAptr[i] = bsr_row_ptr[i];

    hAptr[m] = bsr_row_ptr_end[m - 1];

    // Offload data to device
    ALPHA_INT *dAptr = NULL;
    ALPHA_INT *dAcol = NULL;
    double *dAval    = NULL;
    double *dx       = NULL;
    double *dy       = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
    PRINT_IF_HIP_ERROR(
        hipMalloc((void **)&dAval, sizeof(double) * nnz * bsr_dim * bsr_dim));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dx, sizeof(double) * sizex));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dy, sizeof(double) * sizey));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAcol, bsr_col_index, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dAval, bsr_values, sizeof(double) * nnz * bsr_dim * bsr_dim, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dx, x, sizeof(double) * sizex, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dy, icty, sizeof(double) * sizey, hipMemcpyHostToDevice));

    double halpha = alpha;

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrA;
    alphasparse_dcu_create_mat_descr(&descrA);
    descrA->diag = descr.diag;
    descrA->mode = descr.mode;
    descrA->type = descr.type;

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu bsrmv
        alpha_call_exit(alphasparse_dcu_d_bsrsv_solve(
                            handle, layout, transA, mb, nnzb, &halpha, descrA, dAval, dAptr, dAcol, bsr_dim, NULL, dx, dy, ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO, NULL),
                        "alphasparse_dcu_d_bsrsv_solve");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call alphasparse_dcu bsrmv
            alpha_call_exit(alphasparse_dcu_d_bsrsv_solve(
                                handle, layout, transA, mb, nnzb, &halpha, descrA, dAval, dAptr, dAcol, bsr_dim, NULL, dx, dy, ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO, NULL),
                            "alphasparse_dcu_d_bsrsv_solve");
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    std::cout << time << ",";

    hipMemcpy(icty, dy, sizeof(double) * sizey, hipMemcpyDeviceToHost);
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
    layout     = alpha_args_get_layout(argc, argv);

    alphasparse_index_base_t bsr_index;

    alphasparse_matrix_t coo, bsr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    double *coo_values;
    // read coo
    alpha_read_coo_d(file, &mb, &nb, &nnzb, &coo_row_index, &coo_col_index, &coo_values);

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, mb, nb, nnzb, coo_row_index, coo_col_index, coo_values),
        "alphasparse_d_create_coo");
    // 将稀疏矩阵从coo格式转换成bsr格式
    alpha_call_exit(
        alphasparse_convert_bsr(coo, bsr_dim, layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsr),
        "alphasparse_convert_bsr");
    // 获取bsr格式里的数据
    alpha_call_exit(alphasparse_d_export_bsr(
                        bsr, &bsr_index, &layout, &mb, &nb, &bsr_dim, &bsr_row_ptr, &bsr_row_ptr_end, &bsr_col_index, &bsr_values),
                    "alphasparse_d_export_bsr");
    nnzb = bsr_row_ptr_end[mb - 1];

    sizex = mb * bsr_dim, sizey = nb * bsr_dim;
    if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
        roctransA = rocsparse_operation_transpose;
    } else if (transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        roctransA = rocsparse_operation_conjugate_transpose;
    }

    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        roclayout = rocsparse_direction_row;
    else
        roclayout = rocsparse_direction_column;

    // init x y
    x    = (double *)alpha_malloc(sizex * sizeof(double));
    icty = (double *)alpha_malloc(sizey * sizeof(double));
    rocy = (double *)alpha_malloc(sizey * sizeof(double));

    alpha_fill_random_d(x, 0, sizex);
    alpha_fill_random_d(icty, 1, sizey);
    alpha_fill_random_d(rocy, 1, sizey);
    // memset(x, 0, sizex);
    // memset(icty, 0, sizey);
    // memset(rocy, 0, sizey);

    alpha_trsv_dcu();
    if (check) {
        roc_trsv_dcu();
        // std::cout << "roc_mv end" << std::endl;
        // int len = sizey < 100 ? sizey : 100;
        // for(int i = 0; i < len; i++)
        // {
        //     std::cout << rocy[i] << " " << icty[i] << std::endl;
        // }
        check_d((double *)rocy, sizey, (double *)icty, sizey);
    }
    printf("\n");

    alpha_free(x);
    alpha_free(icty);
    alpha_free(rocy);
    // alpha_free(bsr_row_ptr);
    // alpha_free(bsr_row_ptr_end);
    // alpha_free(bsr_col_index);
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
