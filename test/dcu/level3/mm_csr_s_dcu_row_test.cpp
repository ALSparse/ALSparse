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
struct alpha_matrix_descr descr;
alphasparse_layout_t layout;

alphasparse_matrix_t coo, csr;

ALPHA_INT columns;
// csr format
ALPHA_INT A_rows, A_cols, rnnz;
ALPHA_INT *csr_row_ptr, *csr_row_ptr_end, *csr_col_index;
float *csr_values;

// parms for kernel
float *matB, *matC_ict, *matC_roc;
ALPHA_INT C_rows, C_cols, C_k;
ALPHA_INT B_rows;
ALPHA_INT ldb, ldc;
const float alpha = 2.f;
const float beta = 3.f;

ALPHA_INT warm_up = 5;
ALPHA_INT trials = 10;

#ifdef __PLAIN__
static void alpha_mm_plain() {
  // Start time measurement
  double time = get_time_us();

  for (ALPHA_INT i = 0; i < (warm_up + trials); i++) {
    alpha_call_exit(alphasparse_s_mm_plain(transA, alpha, csr, descr, layout, matB, columns, ldb, beta, matC_roc, ldc),
                  "alphasparse_s_mm_plain");
  }

  time = (get_time_us() - time) / ((trials + warm_up) * 1e3);

  std::cout << time << std::endl;
}
#endif

static void alpha_mm_general() {
  // rocSPARSE handle
  alphasparse_dcu_handle_t handle;
  init_handle(&handle);
  alphasparse_dcu_get_handle(&handle);
  handle->check = true;

  // Generate problem
  ALPHA_INT m = C_rows;
  ALPHA_INT n = C_cols;
  ALPHA_INT k = C_k;
  ALPHA_INT nnz = rnnz;

  ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
  ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz);
  float *hAval = (float *)alpha_malloc(sizeof(float) * nnz);

  for (int i = 0; i < m; i++) hAptr[i] = csr_row_ptr[i];

  hAptr[m] = csr_row_ptr_end[m - 1];

  for (int i = 0; i < nnz; i++) {
    hAcol[i] = csr_col_index[i];
    hAval[i] = csr_values[i];
  }

  // Offload data to device
  ALPHA_INT *dAptr = NULL;
  ALPHA_INT *dAcol = NULL;
  float *dAval = NULL;
  float *dmatB = NULL;
  float *dmatC = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dmatB, sizeof(float) * k * ldb));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dmatC, sizeof(float) * C_rows * ldc));

  PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dAval, hAval, sizeof(float) * nnz, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dmatB, matB, sizeof(float) * k * ldb, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dmatC, matC_roc, sizeof(float) * C_rows * ldc, hipMemcpyHostToDevice));

  float halpha = alpha;
  float hbeta  = beta;

  // Matrix descriptor
  alpha_dcu_matrix_descr_t descrA;
  alphasparse_dcu_create_mat_descr(&descrA);

  // Warm up
  for (int i = 0; i < warm_up; ++i) {
    // Call alphasparse_dcu csrmm
    alphasparse_dcu_s_csrmm(handle, transA, transB, layout, m, n, k, nnz, &halpha,
                           descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta,
                           dmatC, ldc);
    hipDeviceSynchronize();
  }

  // Device synchronization
  hipDeviceSynchronize();

  // Start time measurement
  std::vector<double> times;

  // CSR matrix vector multiplication
  for (int i = 0; i < trials; ++i) {
    double time = get_time_us();
    // Call alphasparse_dcu csrmm
    alphasparse_dcu_s_csrmm(handle, transA, transB, layout, m, n, k, nnz, &halpha,
                            descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta,
                            dmatC, ldc);
    // Device synchronization
    hipDeviceSynchronize();
    time = (get_time_us() - time) / (1e3);
    times.push_back(time);
  }

  // time = (get_time_us() - time) / (trials * 1e3);
  double time = get_avg_time(times);
  std::cout << time << "\n";

  hipMemcpy(matC_roc, dmatC, sizeof(float) * C_rows * ldc,
            hipMemcpyDeviceToHost);

  // Clear up on device
  hipFree(dAptr);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dmatB);
  hipFree(dmatC);

  alphasparse_dcu_destroy_mat_descr(descrA);
  alphasparse_dcu_destory_handle(handle);
}

static void alpha_mm_dcu() {
  // rocSPARSE handle
  alphasparse_dcu_handle_t handle;
  init_handle(&handle);
  alphasparse_dcu_get_handle(&handle);
  handle->check = false;

  // Generate problem
  ALPHA_INT m = C_rows;
  ALPHA_INT n = C_cols;
  ALPHA_INT k = C_k;
  ALPHA_INT nnz = rnnz;

  ALPHA_INT *hAptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (m + 1));
  ALPHA_INT *hAcol = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz);
  float *hAval = (float *)alpha_malloc(sizeof(float) * nnz);

  for (int i = 0; i < m; i++) hAptr[i] = csr_row_ptr[i];

  hAptr[m] = csr_row_ptr_end[m - 1];

  for (int i = 0; i < nnz; i++) {
    hAcol[i] = csr_col_index[i];
    hAval[i] = csr_values[i];
  }

  // Offload data to device
  ALPHA_INT *dAptr = NULL;
  ALPHA_INT *dAcol = NULL;
  float *dAval = NULL;
  float *dmatB = NULL;
  float *dmatC = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dmatB, sizeof(float) * k * ldb));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dmatC, sizeof(float) * C_rows * ldc));

  PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dAval, hAval, sizeof(float) * nnz, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dmatB, matB, sizeof(float) * k * ldb, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dmatC, matC_ict, sizeof(float) * C_rows * ldc, hipMemcpyHostToDevice));

  float halpha = alpha;
  float hbeta  = beta;

  // Matrix descriptor
  alpha_dcu_matrix_descr_t descrA;
  alphasparse_dcu_create_mat_descr(&descrA);

  // Warm up
  for (int i = 0; i < warm_up; ++i) {
    // Call alphasparse_dcu csrmm
    alphasparse_dcu_s_csrmm(handle, transA, transB, layout, m, n, k, nnz, &halpha,
                           descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta,
                           dmatC, ldc);
    hipDeviceSynchronize();
  }

  // Device synchronization
  hipDeviceSynchronize();

  // Start time measurement
  std::vector<double> times;

  // CSR matrix vector multiplication
  for (int i = 0; i < trials; ++i) {
    double time = get_time_us();
    // Call alphasparse_dcu csrmm
    alphasparse_dcu_s_csrmm(handle, transA, transB, layout, m, n, k, nnz, &halpha,
                            descrA, dAval, dAptr, dAcol, dmatB, ldb, &hbeta,
                            dmatC, ldc);
    // Device synchronization
    hipDeviceSynchronize();
    time = (get_time_us() - time) / (1e3);
    times.push_back(time);
  }

  double time = get_avg_time(times);
  std::cout << time << ",";

  hipMemcpy(matC_ict, dmatC, sizeof(float) * C_rows * ldc,
            hipMemcpyDeviceToHost);

  // Clear up on device
  hipFree(dAptr);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dmatB);
  hipFree(dmatC);

  alphasparse_dcu_destroy_mat_descr(descrA);
  alphasparse_dcu_destory_handle(handle);
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  transB = alpha_args_get_transB(argc, argv);
  descr = alpha_args_get_matrix_descrA(argc, argv);
  layout = alpha_args_get_layout(argc, argv);

  alphasparse_index_base_t csr_index;

  ALPHA_INT *coo_row_index, *coo_col_index;
  float *coo_values;
  // read coo
  alpha_read_coo(file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index,
               &coo_values);
  columns = args_get_columns(argc, argv, A_rows);  // 默认C是方阵

  // 创建coo格式稀疏矩阵
  alpha_call_exit(
      alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, A_rows, A_cols,
                              rnnz, coo_row_index, coo_col_index, coo_values),
      "alphasparse_s_create_coo");
  // 将稀疏矩阵从coo格式转换成csr格式
  alpha_call_exit(
      alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
      "alphasparse_convert_csr");
  // 获取csr格式里的数据
  alpha_call_exit(
      alphasparse_s_export_csr(csr, &csr_index, &A_rows, &A_cols, &csr_row_ptr,
                              &csr_row_ptr_end, &csr_col_index, &csr_values),
      "alphasparse_s_export_csr");

  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE && transB == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    C_rows = A_rows;
    C_cols = columns;
    C_k = A_cols;
    B_rows = A_cols;
    
    ldb = columns;
    ldc = columns;
    // ldb = (columns + 3) / 4 * 4;  //按4对齐，float4
    // ldc = (columns + 3) / 4 * 4;
    if (ldb % 128 == 0) {
      ldb = ldb + 16;
      ldc = ldb + 16;
    }
  }
  // printf("ldb: %d\n", ldb);

  // init B C
  matB = (float *)alpha_malloc(B_rows * ldb * sizeof(float));
  matC_ict = (float *)alpha_malloc(C_rows * ldc * sizeof(float));
  matC_roc = (float *)alpha_malloc(C_rows * ldc * sizeof(float));

  alpha_fill_random_s(matB, 0, B_rows * ldb);
  alpha_fill_random_s(matC_ict, 1, C_rows * ldc);
  alpha_fill_random_s(matC_roc, 1, C_rows * ldc);

  alpha_mm_dcu();

  if (check) {
// #ifdef __PLAIN__
    // 注意warm up
    // alpha_mm_plain();
// #endif
    alpha_mm_general();
    // for (int i = 0; i < 100; i++)
    // {
    //     cout << "rocC:" << matC_roc[i] << " ictC:" << matC_ict[i] << endl;
    // }
    check_s((float *)matC_roc, C_rows * ldc, (float *)matC_ict, C_rows * ldc);
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
