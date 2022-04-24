/**
 * @brief ict dcu mm bsr test
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

#include <alphasparse_dcu.h>

const char *file;
bool check;

alphasparse_operation_t transA, transB;
rocsparse_operation roctransA, roctransB;
struct alpha_matrix_descr descr;
alphasparse_layout_t layout;
rocsparse_direction roclayout;

ALPHA_INT columns;
// bsr format
ALPHA_INT A_rowsb, A_colsb, nnzb, bs_r = 2, bs_c = 4;
ALPHA_INT *bsr_row_ptr, *bsr_row_ptr_end, *bsr_col_index;
float *bsr_values;

// parms for kernel
float *matB, *matC_ict, *matC_roc;
ALPHA_INT C_rows, C_cols, C_k;
ALPHA_INT ldb, ldc;
const float alpha = 2.f;
const float beta  = 3.f;

const ALPHA_INT warm_up = 5;
const ALPHA_INT trials  = 10;
const int batch_size    = 1;

// static void roc_mm_dcu() {
//   // rocSPARSE handle
//   rocsparse_handle handle;
//   rocsparse_create_handle(&handle);

//   hipDeviceProp_t devProp;
//   int device_id = 0;

//   hipGetDevice(&device_id);
//   hipGetDeviceProperties(&devProp, device_id);
//   // std:: cout << "Device: " << devProp.name << std:: endl;

//   rocsparse_int m = C_rows / bs_r;
//   rocsparse_int n = C_cols;
//   rocsparse_int k = C_k / bs_c;
//   rocsparse_int nnz = nnzb * bs_r * bs_c;
//   rocsparse_int nnb = nnzb;

//   // Generate problem
//   std::vector<rocsparse_int> hAptr(m + 1);
//   std::vector<rocsparse_int> hAcol(nnb);
//   std::vector<float> hAval(nnz);

//   for (int i = 0; i < m; i++) hAptr[i] = bsr_row_ptr[i];

//   hAptr[m] = bsr_row_ptr_end[m - 1];

//   for (int i = 0; i < nnb; i++) {
//     hAcol[i] = bsr_col_index[i];
//   }
//   for (int i = 0; i < nnz; i++) {
//     hAval[i] = bsr_values[i];
//   }

//   // Offload data to device
//   rocsparse_int *dAptr = NULL;
//   rocsparse_int *dAcol = NULL;
//   float *dAval = NULL;
//   float *dmatB = NULL;
//   float *dmatC = NULL;

//   hipMalloc((void **)&dAptr, sizeof(rocsparse_int) * (m + 1));
//   hipMalloc((void **)&dAcol, sizeof(rocsparse_int) * nnb);
//   hipMalloc((void **)&dAval, sizeof(float) * nnz);
//   hipMalloc((void **)&dmatB, sizeof(float) * C_k * ldb);
//   hipMalloc((void **)&dmatC, sizeof(float) * C_cols * ldc);

//   hipMemcpy(dAptr, hAptr.data(), sizeof(rocsparse_int) * (m + 1),
//             hipMemcpyHostToDevice);
//   hipMemcpy(dAcol, hAcol.data(), sizeof(rocsparse_int) * nnb,
//             hipMemcpyHostToDevice);
//   hipMemcpy(dAval, hAval.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
//   hipMemcpy(dmatB, matB, sizeof(float) * C_k * ldb, hipMemcpyHostToDevice);
//   hipMemcpy(dmatC, matC_roc, sizeof(float) * C_cols * ldc,
//             hipMemcpyHostToDevice);

//   float halpha = alpha;
//   float hbeta = beta;

//   // Matrix descriptor
//   rocsparse_mat_descr descrA;
//   rocsparse_create_mat_descr(&descrA);

//   // Warm up
//   for (int i = 0; i < warm_up; ++i) {
//     // Call rocsparse bsrmm
//     roc_call_exit(
//         rocsparse_sgebsrmm(handle, roclayout, roctransA, roctransB, m, n, k,
//                            nnb, &halpha, descrA, dAval, dAptr, dAcol, bs_r,
//                            bs_c, dmatB, ldb, &hbeta, dmatC, ldc),
//         "rocsparse_sgebsrmm");
//   }

//   // Device synchronization
//   hipDeviceSynchronize();

//   // Start time measurement
//   double time = get_time_us();

//   // cout << "m:" << m << " n:" << n << " k:" << k << " nnz:" << nnz << endl;
//   // cout << "ldb:" << ldb << " ldc" << ldc << endl;
//   // CSR matrix vector multiplication
//   for (int i = 0; i < trials; ++i) {
//     for (int i = 0; i < batch_size; ++i) {
//       // Call rocsparse bsrmm
//       roc_call_exit(
//           rocsparse_sgebsrmm(handle, roclayout, roctransA, roctransB, m, n, k,
//                              nnb, &halpha, descrA, dAval, dAptr, dAcol, bs_r,
//                              bs_c, dmatB, ldb, &hbeta, dmatC, ldc),
//           "rocsparse_sgebsrmm");
//     }

//     // Device synchronization
//     hipDeviceSynchronize();
//   }

//   time = (get_time_us() - time) / (trials * batch_size * 1e3);
//   std::cout << time << std::endl;

//   hipMemcpy(matC_roc, dmatC, sizeof(float) * C_cols * ldc,
//             hipMemcpyDeviceToHost);

//   // Clear up on device
//   hipFree(dAptr);
//   hipFree(dAcol);
//   hipFree(dAval);
//   hipFree(dmatB);
//   hipFree(dmatC);

//   rocsparse_destroy_mat_descr(descrA);
//   rocsparse_destroy_handle(handle);
// }

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
    // std:: cout << "Device: " << devProp.name << std:: endl;

    // Generate problem
    ALPHA_INT m   = C_rows / bs_r;
    ALPHA_INT n   = C_cols;
    ALPHA_INT k   = C_k / bs_c;
    ALPHA_INT nnz = nnzb * bs_r * bs_c;
    ALPHA_INT nnb = nnzb;

    ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
    ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnb);
    float *hAval     = (float *)alpha_malloc(sizeof(float) * nnz);

    for (int i = 0; i < m; i++)
        hAptr[i] = bsr_row_ptr[i];

    hAptr[m] = bsr_row_ptr_end[m - 1];

    for (int i = 0; i < nnz; i++) {
        hAval[i] = bsr_values[i];
    }
    for (int i = 0; i < nnb; i++) {
        hAcol[i] = bsr_col_index[i];
    }

    // Offload data to device
    ALPHA_INT *dAptr = NULL;
    ALPHA_INT *dAcol = NULL;
    float *dAval     = NULL;
    float *dmatB     = NULL;
    float *dmatC     = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnb));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dmatB, sizeof(float) * C_k * ldb));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dmatC, sizeof(float) * C_cols * ldc));

    PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnb, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dAval, hAval, sizeof(float) * nnz, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(
        hipMemcpy(dmatB, matB, sizeof(float) * C_k * ldb, hipMemcpyHostToDevice));
    PRINT_IF_HIP_ERROR(hipMemcpy(dmatC, matC_ict, sizeof(float) * C_cols * ldc, hipMemcpyHostToDevice));

    float halpha = alpha;
    float hbeta  = beta;

    // Matrix descriptor
    alpha_dcu_matrix_descr_t descrA;
    alphasparse_dcu_create_mat_descr(&descrA);

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        // Call alphasparse_dcu bsrmm
        alpha_call_exit(
            alphasparse_dcu_s_gebsrmm(handle, layout, transA, transB, m, n, k, nnb, &halpha, descrA, dAval, dAptr, dAcol, bs_r, bs_c, dmatB, ldb, &hbeta, dmatC, ldc),
            "alphasparse_dcu_s_gebsrmm");
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for (int i = 0; i < trials; ++i) {
        for (int i = 0; i < batch_size; ++i) {
            // Call alphasparse_dcu bsrmm
            alpha_call_exit(
                alphasparse_dcu_s_gebsrmm(handle, layout, transA, transB, m, n, k, nnb, &halpha, descrA, dAval, dAptr, dAcol, bs_r, bs_c, dmatB, ldb, &hbeta, dmatC, ldc),
                "alphasparse_dcu_s_gebsrmm");
        }
        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    std::cout << time << ",";

    hipMemcpy(matC_ict, dmatC, sizeof(float) * C_cols * ldc, hipMemcpyDeviceToHost);

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

    alphasparse_index_base_t bsr_index;

    alphasparse_matrix_t coo, bsr;
    ALPHA_INT *coo_row_index, *coo_col_index;
    float *coo_values;

    // read coo
    alpha_read_coo(file, &A_rowsb, &A_colsb, &nnzb, &coo_row_index, &coo_col_index, &coo_values);
    columns = args_get_columns(argc, argv, A_colsb); // 默认C是方阵

    // 创建coo格式稀疏矩阵
    alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, A_rowsb, A_colsb, nnzb, coo_row_index, coo_col_index, coo_values),
                    "alphasparse_s_create_coo");
    // 将稀疏矩阵从coo格式转换成gebsr格式
    alpha_call_exit(
        alphasparse_convert_gebsr(coo, bs_r, bs_c, layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsr),
        "alphasparse_convert_gebsr");
    // 获取bsr格式里的数据
    alpha_call_exit(
        alphasparse_s_export_gebsr(bsr, &bsr_index, &layout, &A_rowsb, &A_colsb, &bs_r, &bs_c, &bsr_row_ptr, &bsr_row_ptr_end, &bsr_col_index, &bsr_values),
        "alphasparse_s_export_gebsr");
    nnzb = bsr_row_ptr_end[A_rowsb - 1];
    // cout << "create bsr finished" << endl;

    if (layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        roclayout = rocsparse_direction_row;
    else
        roclayout = rocsparse_direction_column;

    if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
        if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            C_rows = A_rowsb * bs_r;
            C_cols = columns;
            C_k    = A_colsb * bs_c;
            ldb    = A_colsb * bs_c;
            ldc    = A_rowsb * bs_r;
        } else // transB, conjB, B转置就用方阵测
        {
            C_rows  = A_rowsb * bs_r;
            C_cols  = A_colsb * bs_c;
            C_k     = A_colsb * bs_c;
            columns = ldb = A_colsb * bs_c;
            ldc           = A_rowsb * bs_r;
        }
    } else // transA, conjA
    {
        if (transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
            C_rows = A_colsb * bs_c;
            C_cols = columns;
            C_k    = A_rowsb * bs_r;
            ldb    = C_cols;
            ldc    = C_rows;
        } else // transB, conjB, B转置就用方阵测
        {
            C_rows  = A_rowsb * bs_r;
            C_cols  = A_colsb * bs_c;
            C_k     = A_rowsb * bs_r;
            columns = ldb = C_cols;
            ldc           = C_rows;
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
    matB     = (float *)alpha_malloc(C_k * ldb * sizeof(float));
    matC_ict = (float *)alpha_malloc(C_cols * ldc * sizeof(float));
    matC_roc = (float *)alpha_malloc(C_cols * ldc * sizeof(float));

    alpha_fill_random_s(matB, 0, C_k * ldb);
    alpha_fill_random_s(matC_ict, 1, C_cols * ldc);
    alpha_fill_random_s(matC_roc, 1, C_cols * ldc);

    alpha_mm_dcu();

    // if (check) {
    //   roc_mm_dcu();
    //   // for (int i = 0; i < C_cols * ldc; i++)
    //   // {
    //   //     if (abs(matC_roc[i] - matC_ict[i]) > 0.0001)
    //   //         cout << i << "rocC:" << matC_roc[i] << " ictC:" << matC_ict[i] <<
    //   //         endl;
    //   // }
    //   check_s((float *)matC_roc, C_cols * ldc, (float *)matC_ict, C_cols * ldc);
    // }
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
