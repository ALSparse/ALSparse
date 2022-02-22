#pragma once

#include "../spmat.h"

// mv
alphasparse_status_t dcu_gemv_z_csr5(   alphasparse_dcu_handle_t handle,
                                        const ALPHA_Complex16 alpha,
                                        const spmat_csr5_z_t *csr5,
                                        alphasparse_dcu_mat_info_t info,
                                        const ALPHA_Complex16 *x,
                                        const ALPHA_Complex16 beta,
                                        ALPHA_Complex16 *y);