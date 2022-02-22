#pragma once

#include "../spmat.h"

// mv
alphasparse_status_t dcu_gemv_c_csr5(   alphasparse_dcu_handle_t handle,
                                        const ALPHA_Complex8 alpha,
                                        const spmat_csr5_c_t *csr5,
                                        alphasparse_dcu_mat_info_t info,
                                        const ALPHA_Complex8 *x,
                                        const ALPHA_Complex8 beta,
                                        ALPHA_Complex8 *y);