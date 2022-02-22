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

#include "rocsparse.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <alpha_spblas_dcu.h>

const char *file;
bool check;
int iter;

// sparse vector
ALPHA_INT nnz = 10000;
ALPHA_INT *alpha_x_idx;
rocsparse_int *roc_x_idx;
float *alpha_x_val, *roc_x_val;
float *y;

// static void roc_gthr() {
//   // rocSPARSE handle
//   rocsparse_handle handle;
//   rocsparse_create_handle(&handle);

//   hipDeviceProp_t devProp;
//   int device_id = 0;

//   hipGetDevice(&device_id);
//   hipGetDeviceProperties(&devProp, device_id);
//   std::cout << "Device: " << devProp.name << std::endl;

//   // Offload data to device
//   rocsparse_int *dx_idx = NULL;
//   float *dx_val = NULL;
//   float *dy = NULL;

//   hipMalloc((void **)&dx_idx, sizeof(rocsparse_int) * nnz);
//   hipMalloc((void **)&dx_val, sizeof(float) * nnz);
//   hipMalloc((void **)&dy, sizeof(float) * nnz * 20);

//   hipMemcpy(dx_idx, roc_x_idx, sizeof(rocsparse_int) * nnz,
//             hipMemcpyHostToDevice);
//   hipMemcpy(dx_val, roc_x_val, sizeof(float) * nnz, hipMemcpyHostToDevice);
//   hipMemcpy(dy, y, sizeof(float) * nnz * 20, hipMemcpyHostToDevice);

//   rocsparse_spvec_descr x;
//   rocsparse_create_spvec_descr(&x, nnz * 20, nnz, (void *)dx_idx,
//                                (void *)dx_val, rocsparse_indextype_i32,
//                                rocsparse_index_base_zero,
//                                rocsparse_datatype_f32_r);
//   rocsparse_dnvec_descr y;
//   rocsparse_create_dnvec_descr(&y, nnz * 20, (void *)dy,
//                                rocsparse_datatype_f32_r);

//   // Call rocsparse csrmv
//   roc_call_exit(rocsparse_gather(handle, y, x), "rocsparse_gather");

//   // Device synchronization
//   hipDeviceSynchronize();

//   hipMemcpy(roc_x_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);

//   // Clear up on device
//   hipFree(dx_val);
//   hipFree(dx_idx);
//   hipFree(dy);
//   rocsparse_destroy_handle(handle);
// }

static void alpha_gthr()
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

    // Offload data to device
    ALPHA_INT *dx_idx = NULL;
    float *dx_val     = NULL;
    float *dy         = NULL;

    hipMalloc((void **)&dx_idx, sizeof(ALPHA_INT) * nnz);
    hipMalloc((void **)&dx_val, sizeof(float) * nnz);
    hipMalloc((void **)&dy, sizeof(float) * nnz * 20);

    hipMemcpy(dx_idx, roc_x_idx, sizeof(ALPHA_INT) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx_val, alpha_x_val, sizeof(float) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dy, y, sizeof(float) * nnz * 20, hipMemcpyHostToDevice);

    alphasparse_dcu_spvec_descr_t x = new _alphasparse_dcu_spvec_descr();

    x->init      = true;
    x->size      = nnz * 20;
    x->nnz       = nnz;
    x->idx_data  = (void *)dx_idx;
    x->val_data  = (void *)dx_val;
    x->idx_type  = ALPHA_SPARSE_DCU_INDEXTYPE_I32;
    x->data_type = ALPHA_SPARSE_DATATYPE_FLOAT;
    x->idx_base  = ALPHA_SPARSE_INDEX_BASE_ZERO;

    alphasparse_dcu_dnvec_descr_t y = new _alphasparse_dcu_dnvec_descr();

    y->init      = true;
    y->size      = nnz * 20;
    y->values    = (void *)dy;
    y->data_type = ALPHA_SPARSE_DATATYPE_FLOAT;

    // Call rocsparse csrmv
    alphasparse_dcu_gather(handle, y, x);

    // Device synchronization
    hipDeviceSynchronize();

    hipMemcpy(alpha_x_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dx_val);
    hipFree(dx_idx);
    hipFree(dy);
    alphasparse_dcu_destory_handle(handle);
}

int main(int argc, const char *argv[])
{
    // args
    args_help(argc, argv);
    file  = args_get_data_file(argc, argv);
    check = args_get_if_check(argc, argv);
    iter  = args_get_iter(argc, argv);

    alpha_x_idx = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * nnz, DEFAULT_ALIGNMENT);
    roc_x_idx   = (rocsparse_int *)alpha_memalign(sizeof(rocsparse_int) * nnz,
                                                DEFAULT_ALIGNMENT);
    alpha_x_val = (float *)alpha_memalign(sizeof(float) * nnz, DEFAULT_ALIGNMENT);
    roc_x_val   = (float *)alpha_memalign(sizeof(float) * nnz, DEFAULT_ALIGNMENT);
    y           = (float *)alpha_memalign(sizeof(float) * nnz * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(y, 0, nnz * 20);
    alpha_fill_random_s(alpha_x_val, 1, nnz);
    alpha_fill_random_s(roc_x_val, 1, nnz);

    for (ALPHA_INT i = 0; i < nnz; i++) {
        alpha_x_idx[i] = i * 20;
        roc_x_idx[i]   = i * 20;
    }

    alpha_gthr();

    // if (check) {
    //   roc_gthr();
    //   check_s(alpha_x_val, nnz, roc_x_val, nnz);
    // }
    printf("\n");

    alpha_free(roc_x_val);
    alpha_free(alpha_x_val);
    alpha_free(roc_x_idx);
    alpha_free(alpha_x_idx);
    alpha_free(y);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
