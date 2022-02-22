#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

__global__ static void
dcu_hyb_gemv(alphasparse_dcu_handle_t handle,
             const ALPHA_Number alpha,
             const alphasparse_dcu_hyb_mat_t hyb,
             const ALPHA_Number* x,
             const ALPHA_Number beta,
             ALPHA_Number* y)
{
    ALPHA_INT ix     = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    ALPHA_INT stride = hipBlockDim_x * hipGridDim_x;
}

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      const ALPHA_Number alpha,
      const alphasparse_dcu_hyb_mat_t hyb,
      const ALPHA_Number* x,
      const ALPHA_Number beta,
      ALPHA_Number* y)
{
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
