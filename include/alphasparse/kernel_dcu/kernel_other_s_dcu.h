#pragma once

#include "../spmat.h"

// mv
alphasparse_status_t dcu_gemv_s_csr5(   alphasparse_dcu_handle_t handle,
                                        const float alpha,
                                        const spmat_csr5_s_t *csr5,
                                        alphasparse_dcu_mat_info_t info,
                                        const float *x,
                                        const float beta,
                                        float *y);