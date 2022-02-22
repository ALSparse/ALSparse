#pragma once

#include "../spmat.h"

alphasparse_status_t dcu_z_doti(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex16 *x_val,
                               const ALPHA_INT *x_ind,
                               const ALPHA_Complex16 *y,
                               ALPHA_Complex16 *result);

alphasparse_status_t dcu_z_dotci(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex16 *x_val,
                                const ALPHA_INT *x_ind,
                                const ALPHA_Complex16 *y,
                                ALPHA_Complex16 *result);

alphasparse_status_t dcu_z_axpyi(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex16 alpha,
                                const ALPHA_Complex16 *x_val,
                                const ALPHA_INT *x_ind,
                                ALPHA_Complex16 *y);

alphasparse_status_t dcu_z_gthr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex16 *y,
                               ALPHA_Complex16 *x_val,
                               const ALPHA_INT *x_ind);

alphasparse_status_t dcu_z_gthrz(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const ALPHA_Complex16 *y,
                                ALPHA_Complex16 *x_val,
                                const ALPHA_INT *x_ind);

alphasparse_status_t dcu_z_sctr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const ALPHA_Complex16 *x_val,
                               const ALPHA_INT *x_ind,
                               ALPHA_Complex16 *y);

alphasparse_status_t dcu_z_axpby(alphasparse_dcu_handle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);