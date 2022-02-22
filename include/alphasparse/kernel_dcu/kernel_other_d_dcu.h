#pragma once

#include "../spmat.h"

// mv
alphasparse_status_t dcu_gemv_d_csr5(   alphasparse_dcu_handle_t handle,
                                        const double alpha,
                                        const spmat_csr5_d_t *csr5,
                                        alphasparse_dcu_mat_info_t info,
                                        const double *x,
                                        const double beta,
                                        double *y);