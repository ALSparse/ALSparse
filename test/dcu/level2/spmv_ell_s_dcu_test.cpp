/**
 * @brief ict dcu mv csr test
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
#include "alphasparse/util/args.h"
#include "alphasparse/util/io.h"
#include "alphasparse/util/check.h"
#include <alphasparse_dcu.h>

const char *file;
bool check;

alphasparse_operation_t transA;
alphasparse_layout_t layout;
struct alpha_matrix_descr descr;

alphasparse_matrix_t coo, csr, ell;

// parms for kernel
float *x, *icty, *refy;
ALPHA_INT sizex, sizey, rm, rk, rnnz;
const float alpha = 2.f;
const float beta = 3.f;

ALPHA_INT *csr_row_ptr, *csr_row_ptr_end, *csr_col_index;
float *csr_values;

const ALPHA_INT warm_up = 5;
ALPHA_INT trials = 10;

static void ref_csrmv_dcu() {
  // rocSPARSE handle
  alphasparse_dcu_handle_t handle;
  init_handle(&handle);
  alphasparse_dcu_get_handle(&handle);

  // Generate problem
  ALPHA_INT m = rm;
  ALPHA_INT n = rk;
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
  float *dx = NULL;
  float *dy = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAptr, sizeof(ALPHA_INT) * (m + 1)));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(ALPHA_INT) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dx, sizeof(float) * sizex));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dy, sizeof(float) * sizey));

  PRINT_IF_HIP_ERROR(hipMemcpy(dAptr, hAptr, sizeof(ALPHA_INT) * (m + 1),
                               hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(
      hipMemcpy(dAcol, hAcol, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(
      hipMemcpy(dAval, hAval, sizeof(float) * nnz, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(
      hipMemcpy(dx, x, sizeof(float) * sizex, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(
      hipMemcpy(dy, refy, sizeof(float) * sizey, hipMemcpyHostToDevice));

  float halpha = alpha;
  float hbeta = beta;

  // Matrix descriptor
  alpha_dcu_matrix_descr_t descrA;
  alphasparse_dcu_create_mat_descr(&descrA);

  // Warm up
  for (int i = 0; i < warm_up; ++i) {
    // Call alphasparse_dcu csrmv
    alpha_call_exit(
        alphasparse_dcu_s_csrmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m, n,
                               nnz, &halpha, descrA, dAval, dAptr, dAcol, nullptr,
                               dx, &hbeta, dy),
        "alphasparse_dcu_s_csrmv");
    hipDeviceSynchronize();
  }

  // Start time measurement
  std::vector<double> times;

  // CSR matrix vector multiplication
  for (int i = 0; i < trials; ++i) {
    double time = get_time_us();
    // Call alphasparse_dcu csrmv
    alpha_call_exit(
        alphasparse_dcu_s_csrmv(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, m,
                                n, nnz, &halpha, descrA, dAval, dAptr, dAcol,
                                nullptr, dx, &hbeta, dy),
        "alphasparse_dcu_s_csrmv");
    // Device synchronization
    hipDeviceSynchronize();
    time = (get_time_us() - time) / (1e3);
    times.push_back(time);
  }

  double time = get_avg_time(times);
  double bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + sizeof(ALPHA_INT) * (m + 1 + nnz) / time / 1e6; 
  double gflops    = static_cast<double>(2 * nnz) / time / 1e6;

  std::cout << time;
  // std::cout << " " << bandwidth;
  // std::cout << " " << gflops;
  std::cout << "\n";

  hipMemcpy(refy, dy, sizeof(float) * sizey, hipMemcpyDeviceToHost);

  // Clear up on device
  hipFree(dAptr);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dx);
  hipFree(dy);

  alphasparse_dcu_destroy_mat_descr(descrA);
  alphasparse_dcu_destory_handle(handle);
}

static void alpha_mv_dcu() {
  // rocSPARSE handle
  alphasparse_dcu_handle_t handle;
  init_handle(&handle);
  alphasparse_dcu_get_handle(&handle);

  // Generate problem
  ALPHA_INT m = rm;
  ALPHA_INT n = rk;
  ALPHA_INT nnz = rnnz;

  // Offload data to device
  float *dx = NULL;
  float *dy = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dx, sizeof(float) * sizex));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dy, sizeof(float) * sizey));

  PRINT_IF_HIP_ERROR(hipMemcpy(dx, x, sizeof(float) * sizex, hipMemcpyHostToDevice));
  PRINT_IF_HIP_ERROR(hipMemcpy(dy, icty, sizeof(float) * sizey, hipMemcpyHostToDevice));

  // Matrix descriptor
  alpha_dcu_matrix_descr_t descrA;
  alphasparse_dcu_create_mat_descr(&descrA);

  // Warm up
  for (int i = 0; i < warm_up; ++i) {
    // Call alphasparse_dcu csrmv
    alpha_call_exit(alphasparse_s_mv_dcu(handle, transA, &alpha, ell, descrA, dx, &beta, dy), 
      "alphasparse_s_mv_dcu");
  }

  // Device synchronization
  hipDeviceSynchronize();

  // Start time measurement
  double time = get_time_us();

  // CSR matrix vector multiplication
  for (int i = 0; i < trials; ++i) {
    // Call alphasparse_dcu csrmv
    alpha_call_exit(alphasparse_s_mv_dcu(handle, transA, &alpha, ell, descrA, dx, &beta, dy), 
      "alphasparse_s_mv_dcu");
    hipDeviceSynchronize();
  }

  time = (get_time_us() - time) / (trials * 1e3);
  double bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) + sizeof(ALPHA_INT) * (m + 1 + nnz) / time / 1e6; 
  double gflops    = static_cast<double>(2 * nnz) / time / 1e6;

  std::cout << time;
  // std::cout << " " << bandwidth;
  // std::cout << " " << gflops;
  std::cout << ",";

  hipMemcpy(icty, dy, sizeof(float) * sizey, hipMemcpyDeviceToHost);

  // Clear up on device
  hipFree(dx);
  hipFree(dy);

  alphasparse_dcu_destroy_mat_descr(descrA);
  alphasparse_dcu_destory_handle(handle);
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_index_base_t csr_index;

  ALPHA_INT *coo_row_index, *coo_col_index;
  float *coo_values;
  // read coo
  alpha_read_coo(file, &rm, &rk, &rnnz, &coo_row_index, &coo_col_index,
               &coo_values);

  // create coo
  alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, rm, rk, rnnz,coo_row_index, coo_col_index, coo_values),
      "alphasparse_s_create_coo");
  
  // coo2csr
  alpha_call_exit(alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
      "alphasparse_convert_csr");
  
  // get csr data
  alpha_call_exit(alphasparse_s_export_csr(csr, &csr_index, &rm, &rk, &csr_row_ptr, &csr_row_ptr_end, &csr_col_index, &csr_values),
      "alphasparse_s_export_csr");

  // coo2ell
  alpha_call_exit(alphasparse_convert_ell(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &ell),
      "alphasparse_convert_ell");
  
  // convert csr host ptr to device ptr
  alpha_call_exit(host2device_s_ell(ell), "host2device_s_ell");

  sizex = rk, sizey = rm;

  // init x y
  x = (float *)alpha_malloc(sizex * sizeof(float));
  icty = (float *)alpha_malloc(sizey * sizeof(float));
  refy = (float *)alpha_malloc(sizey * sizeof(float));

  alpha_fill_random_s(x, 0, sizex);
  alpha_fill_random_s(icty, 1, sizey);
  alpha_fill_random_s(refy, 1, sizey);

  alpha_mv_dcu();

  if (check) {
    ref_csrmv_dcu();

    check_s((float *)refy, sizey, (float *)icty, sizey);
  }
  printf("\n");

  alpha_call_exit(alphasparse_destroy(csr), "alphasparse_destroy");
  alpha_call_exit(alphasparse_destroy(coo), "alphasparse_destroy");

  return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
