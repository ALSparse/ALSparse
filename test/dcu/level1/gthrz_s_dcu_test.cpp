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

#include <alphasparse_dcu.h>

const char *file;
bool check;
int iter;

// sparse vector
ALPHA_INT nnz = 10000;
ALPHA_INT *alpha_x_idx;
rocsparse_int *roc_x_idx;
float *alpha_x_val, *roc_x_val;
float *alpha_y, *roc_y;

static void roc_gthrz()
{
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    std::cout << "Device: " << devProp.name << std::endl;

    // Offload data to device
    rocsparse_int *dx_idx = NULL;
    float *dx_val         = NULL;
    float *dy             = NULL;

    hipMalloc((void **)&dx_idx, sizeof(rocsparse_int) * nnz);
    hipMalloc((void **)&dx_val, sizeof(float) * nnz);
    hipMalloc((void **)&dy, sizeof(float) * nnz * 20);

    hipMemcpy(dx_idx, roc_x_idx, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx_val, roc_x_val, sizeof(float) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dy, roc_y, sizeof(float) * nnz * 20, hipMemcpyHostToDevice);

    // Call rocsparse csrmv
    roc_call_exit(rocsparse_sgthrz(handle, nnz, dy, dx_val, dx_idx, rocsparse_index_base_zero),
                  "rocsparse_sgthrz");

    // Device synchronization
    hipDeviceSynchronize();

    hipMemcpy(roc_x_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);
    hipMemcpy(roc_y, dx_val, sizeof(float) * nnz * 20, hipMemcpyDeviceToHost);

    // Clear up on device
    hipFree(dx_val);
    hipFree(dx_idx);
    hipFree(dy);
    rocsparse_destroy_handle(handle);
}

static void alpha_gthrz()
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
    hipMemcpy(dy, alpha_y, sizeof(float) * nnz * 20, hipMemcpyHostToDevice);

    // Call rocsparse csrmv
    alphasparse_dcu_s_gthrz(handle, nnz, dy, dx_val, dx_idx, ALPHA_SPARSE_INDEX_BASE_ZERO);

    // Device synchronization
    hipDeviceSynchronize();

    hipMemcpy(alpha_x_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);
    hipMemcpy(alpha_y, dx_val, sizeof(float) * nnz * 20, hipMemcpyDeviceToHost);

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
    alpha_y     = (float *)alpha_memalign(sizeof(float) * nnz * 20, DEFAULT_ALIGNMENT);
    roc_y       = (float *)alpha_memalign(sizeof(float) * nnz * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(alpha_y, 0, nnz * 20);
    alpha_fill_random_s(roc_y, 0, nnz * 20);
    alpha_fill_random_s(alpha_x_val, 1, nnz);
    alpha_fill_random_s(roc_x_val, 1, nnz);

    for (ALPHA_INT i = 0; i < nnz; i++) {
        alpha_x_idx[i] = i * 20;
        roc_x_idx[i]   = i * 20;
    }

    alpha_gthrz();

    if (check) {
        roc_gthrz();
        check_s(alpha_x_val, nnz, roc_x_val, nnz);
        check_s(alpha_y, nnz * 20, roc_y, nnz * 20);
    }
    printf("\n");

    alpha_free(roc_x_val);
    alpha_free(alpha_x_val);
    alpha_free(roc_x_idx);
    alpha_free(alpha_x_idx);
    alpha_free(roc_y);
    alpha_free(alpha_y);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
