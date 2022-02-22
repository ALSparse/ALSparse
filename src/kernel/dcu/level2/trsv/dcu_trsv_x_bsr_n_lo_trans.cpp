#include <hip/hip_runtime.h>

#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "alphasparse/kernel_dcu.h"

alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      alphasparse_layout_t dir,
      ALPHA_INT mb,
      ALPHA_INT nnzb,
      const ALPHA_Number alpha,
      const ALPHA_Number *bsr_val,
      const ALPHA_INT *bsr_row_ptr,
      const ALPHA_INT *bsr_col_ind,
      ALPHA_INT bsr_dim,
      alphasparse_dcu_mat_info_t info,
      const ALPHA_Number *x,
      ALPHA_Number *y,
      alphasparse_dcu_solve_policy_t policy,
      void *temp_buffer)
{
    return dcu_trsv_bsr_n_hi(handle, dir, mb, nnzb, alpha, bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, y, policy, temp_buffer);
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
