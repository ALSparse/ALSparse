#pragma once

#include "../spmat.h"

alphasparse_status_t dcu_d_doti(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const double *x_val,
                               const ALPHA_INT *x_ind,
                               const double *y,
                               double *result);

alphasparse_status_t dcu_d_axpyi(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const double alpha,
                                const double *x_val,
                                const ALPHA_INT *x_ind,
                                double *y);

alphasparse_status_t dcu_d_gthr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const double *y,
                               double *x_val,
                               const ALPHA_INT *x_ind);

alphasparse_status_t dcu_d_gthrz(alphasparse_dcu_handle_t handle,
                                ALPHA_INT nnz,
                                const double *y,
                                double *x_val,
                                const ALPHA_INT *x_ind);

alphasparse_status_t dcu_d_roti(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               double *x_val,
                               const ALPHA_INT *x_ind,
                               double *y,
                               const double *c,
                               const double *s);

alphasparse_status_t dcu_d_sctr(alphasparse_dcu_handle_t handle,
                               ALPHA_INT nnz,
                               const double *x_val,
                               const ALPHA_INT *x_ind,
                               double *y);

alphasparse_status_t dcu_d_axpby(alphasparse_dcu_handle_t handle,
                                const void *alpha,
                                const alphasparse_dcu_spvec_descr_t x,
                                const void *beta,
                                alphasparse_dcu_dnvec_descr_t y);