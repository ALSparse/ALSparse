#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

const ALPHA_INT threadPerBlock = 256;

__global__ static void
doti(ALPHA_INT nnz,
     const ALPHA_Number *x_val,
     const ALPHA_INT *x_ind,
     const ALPHA_Number *y,
     ALPHA_Number *result)
{
    int idx      = threadIdx.x + blockIdx.x * blockDim.x;
    int stride   = gridDim.x * blockDim.x;
    int cacheidx = threadIdx.x;

    __shared__ ALPHA_Number cache[threadPerBlock]; // block内线程共享__shared__，注意这里没有初始化

    ALPHA_Number tmp;
    alpha_setzero(tmp);

    ALPHA_Number x_val_conj;
    for (int i = idx; i < nnz; i += stride) // 每个线程先求和自己可以接触到的数
    {
        alpha_conj(x_val_conj, x_val[i]);
        alpha_madde(tmp, x_val_conj, y[x_ind[i]]);
        // tmp += x_val[i] * y[x_ind[i]];
    }
    cache[cacheidx] = tmp;
    __syncthreads();

    //规约
    ALPHA_INT i = threadPerBlock / 2;
    while (i != 0) {
        if (cacheidx < i) // 只需要线程号小于i的线程参与计算
        {
            //cache[cacheidx] += cache[cacheidx + i]; // 两两求和
            alpha_adde(cache[cacheidx], cache[cacheidx + i]);
        }
        i /= 2; // 循环变量
        __syncthreads();
    }
    if (cacheidx == 0) // 块内0号线程提交块内规约结果
        result[blockIdx.x] = cache[0];
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      ALPHA_INT nnz,
      const ALPHA_Number *x_val,
      const ALPHA_INT *x_ind,
      const ALPHA_Number *y,
      ALPHA_Number *result)
{
    const int blockPerGrid = min(32, (threadPerBlock + nnz - 1) / threadPerBlock);

    ALPHA_Number *dev_part_c, *part_c;

    part_c = (ALPHA_Number *)malloc(sizeof(ALPHA_Number) * blockPerGrid);
    hipMalloc((void **)&dev_part_c, sizeof(ALPHA_Number) * blockPerGrid);

    hipLaunchKernelGGL(doti, dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream, nnz, x_val, x_ind, y, dev_part_c);

    hipMemcpy(part_c, dev_part_c, sizeof(ALPHA_Number) * blockPerGrid, hipMemcpyDeviceToHost);

    alpha_setzero(*result);
    for (int i = 0; i < blockPerGrid; i++) {
        alpha_adde(*result, part_c[i]);
        // *result += part_c[i];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
