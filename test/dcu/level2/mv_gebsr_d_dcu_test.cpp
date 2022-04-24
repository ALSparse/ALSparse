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

// #include "rocsparse_init.hpp"
// #include "rocsparse_random.hpp"
// #include "utility.hpp"

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

// gebsr format
ALPHA_INT rm, rk, nnzb, blk_row = 2, blk_col = 4;
ALPHA_INT *gebsr_row_ptr, *gebsr_row_ptr_end, *gebsr_col_index;
double *gebsr_values;
rocsparse_direction layout_roc;

// parms for kernel
double *x;
double *icty;
double *rocy;
ALPHA_INT sizex, sizey;
const double alpha = 2.f;
const double beta  = 3.f;

const ALPHA_INT warm_up = 5;
const ALPHA_INT trials  = 10;
const int batch_size    = 1;

// static void roc_mv_dcu() {
//   // rocSPARSE handle
//   rocsparse_handle handle;
//   rocsparse_create_handle(&handle);

//   hipDeviceProp_t devProp;
//   int device_id = 0;

//   hipGetDevice(&device_id);
//   hipGetDeviceProperties(&devProp, device_id);
//   // std::cout << "Device: " << devProp.name << std::endl;

//   rocsparse_int m = rm;
//   rocsparse_int n = rk;
//   rocsparse_int nnz = nnzb * blk_col * blk_row;

//   // Generate problem
//   std::vector<rocsparse_int> hAptr(m + 1);
//   std::vector<rocsparse_int> hAcol(nnzb);
//   std::vector<double> hAval(nnz);

//   for (int i = 0; i < m; i++) hAptr[i] = gebsr_row_ptr[i];
//   hAptr[m] = gebsr_row_ptr_end[m - 1];

//   for (int i = 0; i < nnz; i++) {
//     hAval[i] = gebsr_values[i];
//   }
//   for (int i = 0; i < nnzb; i++) {
//     hAcol[i] = gebsr_col_index[i];
//   }

//   // Offload data to device
//   rocsparse_int *dAptr = NULL;
//   rocsparse_int *dAcol = NULL;
//   double *dAval = NULL;
//   double *dx = NULL;
//   double *dy = NULL;

//   hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
//   hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnzb);
//   hipMalloc((void **)&dAval, sizeof(double) * nnz);
//   hipMalloc((void **)&dx, sizeof(double) * sizex);
//   hipMalloc((void **)&dy, sizeof(double) * sizey);

//   hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1),
//             hipMemcpyHostToDevice);
//   hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnzb,
//             hipMemcpyHostToDevice);
//   hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
//   hipMemcpy(dx, x, sizeof(double) * sizex, hipMemcpyHostToDevice);
//   hipMemcpy(dy, rocy, sizeof(double) * sizey, hipMemcpyHostToDevice);

//   double halpha = alpha;
//   double hbeta = beta;

//   // Matrix descriptor
//   rocsparse_mat_descr descrA;
//   rocsparse_create_mat_descr(&descrA);

//   // Warm up
//   for (int i = 0; i < warm_up; ++i) {
//     // Call rocsparse bsrmv
//     roc_call_exit(
//         rocsparse_dgebsrmv(handle, layout_roc, rocsparse_operation_none, m, n,
//                            nnzb, &halpha, descrA, dAval, dAptr, dAcol, blk_row,
//                            blk_col, dx, &hbeta, dy),
//         "rocsparse_dgebsrmv");
//   }

//   // Device synchronization
//   hipDeviceSynchronize();

//   // Start time measurement
//   double time = get_time_us();

//   // bsr matrix vector multiplication
//   for (int i = 0; i < trials; ++i) {
//     for (int i = 0; i < batch_size; ++i) {
//       // Call rocsparse bsrmv
//       roc_call_exit(
//           rocsparse_dgebsrmv(handle, layout_roc, rocsparse_operation_none, m, n,
//                              nnzb, &halpha, descrA, dAval, dAptr, dAcol,
//                              blk_row, blk_col, dx, &hbeta, dy),
//           "rocsparse_dgebsrmv");
//     }

//     // Device synchronization
//     hipDeviceSynchronize();
//   }

//   time = (get_time_us() - time) / (trials * batch_size * 1e3);
//   std::cout << time << std::endl;

//   hipMemcpy(rocy, dy, sizeof(double) * sizey, hipMemcpyDeviceToHost);

//   // Clear up on device
//   hipFree(dAptr);
//   hipFree(dAcol);
//   hipFree(dAval);
//   hipFree(dx);
//   hipFree(dy);

//   rocsparse_destroy_mat_descr(descrA);
//   rocsparse_destroy_handle(handle);
// }

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
    ALPHA_INT m   = rm;
    ALPHA_INT n   = rk;
    ALPHA_INT nnz = nnzb * blk_row * blk_col;

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
    ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnzb);
    double *hAval    = (double *)alpha_malloc(sizeof(double) * nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = gebsr_row_ptr[i];
    hAptr[m] = gebsr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAval[i] = gebsr_values[i];
    }
    for (int i = 0; i < nnzb; i++) {
        hAcol[i] = gebsr_col_index[i];
    }

    // Offload data to device
    ALPHA_INT *dAptr = NULL;
    ALPHA_INT *dAcol = NULL;
    double *dAval    = NULL;
    double *dx       = NULL;
    double *dy       = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnzb));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(double) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dx, sizeof(double) * sizex));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dy, sizeof(double) * sizey));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnzb, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice));
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
        // Call alphasparse_dcu bsrmv
        alpha_call_exit(alphasparse_dcu_d_gebsrmv(
                            handle, layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, nnzb, &halpha, descrA, dAval, dAptr, dAcol, blk_row, blk_col, dx, &hbeta, dy),
                        "alphasparse_dcu_d_gebsrmv");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // bsr matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call alphasparse_dcu bsrmv
            alpha_call_exit(alphasparse_dcu_d_gebsrmv(
                                handle, layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n, nnzb, &halpha, descrA, dAval, dAptr, dAcol, blk_row, blk_col, dx, &hbeta, dy),
                            "alphasparse_dcu_d_gebsrmv");
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

    alphasparse_index_base_t gebsr_index;

    alphasparse_matrix_t coo, gebsr;
    ALPHA_INT *coo_row_index, *coo_col_index, nnz;
    double *coo_values;
    // read coo
    alpha_read_coo_d(file, &rm, &rk, &nnz, &coo_row_index, &coo_col_index, &coo_values);

    // 创建coo格式稀疏矩阵
    alpha_call_exit(
        alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, rm, rk, nnz, coo_row_index, coo_col_index, coo_values),
        "alphasparse_d_create_coo");
    // 将稀疏矩阵从coo格式转换成bsr格式
    alpha_call_exit(
        alphasparse_convert_gebsr(coo, blk_row, blk_col, layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &gebsr),
        "alphasparse_convert_gebsr");
    // 获取bsr格式里的数据
    alpha_call_exit(alphasparse_d_export_gebsr(gebsr, &gebsr_index, &layout, &rm, &rk, &blk_row, &blk_col, &gebsr_row_ptr, &gebsr_row_ptr_end, &gebsr_col_index, &gebsr_values),
                    "alphasparse_d_export_gebsr");

    nnzb  = gebsr_row_ptr_end[rm - 1];
    sizex = rk * blk_col, sizey = rm * blk_row;
    // if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE || transA ==
    // ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    // {
    //     sizex = rm * blk_row;
    //     sizey = rk * blk_col;
    // }
    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
        layout_roc = rocsparse_direction_row;
    } else {
        layout_roc = rocsparse_direction_column;
    }

    // init x y
    x    = (double *)alpha_malloc(sizex * sizeof(double));
    icty = (double *)alpha_malloc(sizey * sizeof(double));
    rocy = (double *)alpha_malloc(sizey * sizeof(double));

    alpha_fill_random_d(x, 0, sizex);
    alpha_fill_random_d(icty, 1, sizey);
    alpha_fill_random_d(rocy, 1, sizey);

    alpha_mv_dcu();

    // if (check) {
    //   roc_mv_dcu();
    //   check_d((double *)rocy, sizey, (double *)icty, sizey);
    // }
    printf("\n");

    alpha_free(x);
    alpha_free(icty);
    alpha_free(rocy);
    alpha_free(coo_row_index);
    alpha_free(coo_col_index);
    alpha_free(coo_values);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
