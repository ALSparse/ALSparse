#pragma once

#include "../spmat.h"

alphasparse_status_t dcu_s_doti(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const float *x_val,
                               const ALPHA_INT *x_ind,
                               const float *y,
                               float *result);

alphasparse_status_t dcu_s_axpyi(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const float alpha,
                                const float *x_val,
                                const ALPHA_INT *x_ind,
                                float *y);

alphasparse_status_t dcu_s_gthr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const float *y,
                               float *x_val,
                               const ALPHA_INT *x_ind);

alphasparse_status_t dcu_s_gthrz(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const float *y,
                                float *x_val,
                                const ALPHA_INT *x_ind);

alphasparse_status_t dcu_s_roti(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               float *x_val,
                               const ALPHA_INT *x_ind,
                               float *y,
                               const float *c,
                               const float *s);

alphasparse_status_t dcu_s_sctr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const float *x_val,
                               const ALPHA_INT *x_ind,
                               float *y);

alphasparse_status_t dcu_s_axpby(alphasparse_dcu_handle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);