#include <hip/hip_runtime.h>

#include "alphasparse/spmat.h"
#include "alphasparse/handle.h"
#include "alphasparse/compute.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/common_dcu.h"

// CSR5 SpMV kernel
// see paper by W. Liu and B. Vinter. (2015).
// "CSR5: An Efficient Storage Format for Cross-Platform
//  Sparse Matrix-Vector Multiplication".
// 29th ACM International Conference on Supercomputing (ICS15). pp. 339-350.
//
// 

#include <hip/hip_runtime.h>


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
alphasparse_status_t
ONAME(alphasparse_dcu_handle_t handle,
      const ALPHA_Number alpha,
      const ALPHA_SPMAT_CSR5 *csr5,
      alphasparse_dcu_mat_info_t info,
      const ALPHA_Number *dx,
      const ALPHA_Number beta,
      ALPHA_Number *dy)
{

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */
