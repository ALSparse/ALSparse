#pragma once

#include "../spmat.h"

alphasparse_status_t dcu_c_doti(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex8 *x_val,
                               const ALPHA_INT *x_ind,
                               const ALPHA_Complex8 *y,
                               ALPHA_Complex8 *result);

alphasparse_status_t dcu_c_dotci(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex8 *x_val,
                                const ALPHA_INT *x_ind,
                                const ALPHA_Complex8 *y,
                                ALPHA_Complex8 *result);

alphasparse_status_t dcu_c_axpyi(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex8 alpha,
                                const ALPHA_Complex8 *x_val,
                                const ALPHA_INT *x_ind,
                                ALPHA_Complex8 *y);

alphasparse_status_t dcu_c_gthr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex8 *y,
                               ALPHA_Complex8 *x_val,
                               const ALPHA_INT *x_ind);

alphasparse_status_t dcu_c_gthrz(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex8 *y,
                                ALPHA_Complex8 *x_val,
                                const ALPHA_INT *x_ind);

alphasparse_status_t dcu_c_sctr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex8 *x_val,
                               const ALPHA_INT *x_ind,
                               ALPHA_Complex8 *y);

alphasparse_status_t dcu_c_axpby(alphasparse_dcu_handle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);