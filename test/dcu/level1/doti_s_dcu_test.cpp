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
ALPHA_INT nnz;
ALPHA_INT *alpha_x_idx;
rocsparse_int *roc_x_idx;
float *x_val, *y;

float roc_res = 1., alpha_res = 2.;

ALPHA_INT idx_n = 10000;

static void roc_doti()
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

    hipMalloc((void **)&dx_idx, sizeof(rocsparse_int) * idx_n);
    hipMalloc((void **)&dx_val, sizeof(float) * idx_n);
    hipMalloc((void **)&dy, sizeof(float) * idx_n * 20);

    hipMemcpy(dx_idx, roc_x_idx, sizeof(rocsparse_int) * idx_n, hipMemcpyHostToDevice);
    hipMemcpy(dx_val, x_val, sizeof(float) * idx_n, hipMemcpyHostToDevice);
    hipMemcpy(dy, y, sizeof(float) * idx_n * 20, hipMemcpyHostToDevice);

    // Call rocsparse csrmv
    roc_call_exit(rocsparse_sdoti(handle, idx_n, dx_val, dx_idx, dy, &roc_res, rocsparse_index_base_zero),
                  "rocsparse_sdoti");

    // Device synchronization
    hipDeviceSynchronize();

    // Clear up on device
    hipFree(dx_val);
    hipFree(dx_idx);
    hipFree(dy);
    rocsparse_destroy_handle(handle);
}

static void alpha_doti()
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

    hipMalloc((void **)&dx_idx, sizeof(ALPHA_INT) * idx_n);
    hipMalloc((void **)&dx_val, sizeof(float) * idx_n);
    hipMalloc((void **)&dy, sizeof(float) * idx_n * 20);

    hipMemcpy(dx_idx, roc_x_idx, sizeof(ALPHA_INT) * idx_n, hipMemcpyHostToDevice);
    hipMemcpy(dx_val, x_val, sizeof(float) * idx_n, hipMemcpyHostToDevice);
    hipMemcpy(dy, y, sizeof(float) * idx_n * 20, hipMemcpyHostToDevice);

    // Call rocsparse csrmv
    alphasparse_dcu_s_doti(handle, idx_n, dx_val, dx_idx, dy, &alpha_res, ALPHA_SPARSE_INDEX_BASE_ZERO);

    // Device synchronization
    hipDeviceSynchronize();

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

    alpha_x_idx =
        (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * idx_n, DEFAULT_ALIGNMENT);
    roc_x_idx = (rocsparse_int *)alpha_memalign(sizeof(rocsparse_int) * idx_n,
                                                DEFAULT_ALIGNMENT);
    x_val     = (float *)alpha_memalign(sizeof(float) * idx_n, DEFAULT_ALIGNMENT);
    y         = (float *)alpha_memalign(sizeof(float) * idx_n * 20, DEFAULT_ALIGNMENT);

    alpha_fill_random_s(y, 1, idx_n * 20);
    alpha_fill_random_s(x_val, 1, idx_n);

    for (ALPHA_INT i = 0; i < idx_n; i++) {
        alpha_x_idx[i] = i * 20;
        roc_x_idx[i]   = i * 20;
    }

    alpha_doti();

    // if (check) {
    //     roc_doti();
    //     bool status = fabs(alpha_res - roc_res) > 1e-2;
    //     if (!status) {
    //         fprintf(stderr, "doti_s correct, %f\n", fabs(alpha_res - roc_res));
    //         fprintf(stderr, "roc : %f, ict : %f\n", roc_res, alpha_res);
    //     } else {
    //         fprintf(stderr, "doti_s error\n");
    //         fprintf(stderr, "roc : %f, ict : %f\n", roc_res, alpha_res);
    //     }
    // }
    printf("\n");

    alpha_free(x_val);
    alpha_free(roc_x_idx);
    alpha_free(alpha_x_idx);
    return 0;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
