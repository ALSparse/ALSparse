#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "spmat.h"
#include "spdef.h"
#include "types.h"

alphasparse_status_t alphasparse_dcu_s_axpyi(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             const float *alpha,
                                             const float *x_val,
                                             const ALPHA_INT *x_ind,
                                             float *y,
                                             alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_d_axpyi(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             const double *alpha,
                                             const double *x_val,
                                             const ALPHA_INT *x_ind,
                                             double *y,
                                             alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_c_axpyi(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 *alpha,
                                             const ALPHA_Complex8 *x_val,
                                             const ALPHA_INT *x_ind,
                                             ALPHA_Complex8 *y,
                                             alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_z_axpyi(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex16 *alpha,
                                             const ALPHA_Complex16 *x_val,
                                             const ALPHA_INT *x_ind,
                                             ALPHA_Complex16 *y,
                                             alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_s_doti(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const float *x_val,
                                            const ALPHA_INT *x_ind,
                                            const float *y,
                                            float *result,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_d_doti(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const double *x_val,
                                            const ALPHA_INT *x_ind,
                                            const double *y,
                                            double *result,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_c_doti(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const ALPHA_Complex8 *x_val,
                                            const ALPHA_INT *x_ind,
                                            const ALPHA_Complex8 *y,
                                            ALPHA_Complex8 *result,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_z_doti(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const ALPHA_Complex16 *x_val,
                                            const ALPHA_INT *x_ind,
                                            const ALPHA_Complex16 *y,
                                            ALPHA_Complex16 *result,
                                            alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_c_dotci(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 *x_val,
                                             const ALPHA_INT *x_ind,
                                             const ALPHA_Complex8 *y,
                                             ALPHA_Complex8 *result,
                                             alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_z_dotci(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex16 *x_val,
                                             const ALPHA_INT *x_ind,
                                             const ALPHA_Complex16 *y,
                                             ALPHA_Complex16 *result,
                                             alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_s_gthr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const float *y,
                                            float *x_val,
                                            const ALPHA_INT *x_ind,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_d_gthr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const double *y,
                                            double *x_val,
                                            const ALPHA_INT *x_ind,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_c_gthr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const ALPHA_Complex8 *y,
                                            ALPHA_Complex8 *x_val,
                                            const ALPHA_INT *x_ind,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_z_gthr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const ALPHA_Complex16 *y,
                                            ALPHA_Complex16 *x_val,
                                            const ALPHA_INT *x_ind,
                                            alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_s_gthrz(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             float *y,
                                             float *x_val,
                                             const ALPHA_INT *x_ind,
                                             alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_d_gthrz(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             double *y,
                                             double *x_val,
                                             const ALPHA_INT *x_ind,
                                             alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_c_gthrz(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             ALPHA_Complex8 *y,
                                             ALPHA_Complex8 *x_val,
                                             const ALPHA_INT *x_ind,
                                             alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_z_gthrz(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT nnz,
                                             ALPHA_Complex16 *y,
                                             ALPHA_Complex16 *x_val,
                                             const ALPHA_INT *x_ind,
                                             alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_s_roti(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            float *x_val,
                                            const ALPHA_INT *x_ind,
                                            float *y,
                                            const float *c,
                                            const float *s,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_d_roti(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            double *x_val,
                                            const ALPHA_INT *x_ind,
                                            double *y,
                                            const double *c,
                                            const double *s,
                                            alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_s_sctr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const float *x_val,
                                            const ALPHA_INT *x_ind,
                                            float *y,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_d_sctr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const double *x_val,
                                            const ALPHA_INT *x_ind,
                                            double *y,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_c_sctr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const ALPHA_Complex8 *x_val,
                                            const ALPHA_INT *x_ind,
                                            ALPHA_Complex8 *y,
                                            alphasparse_index_base_t idx_base);

alphasparse_status_t alphasparse_dcu_z_sctr(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT nnz,
                                            const ALPHA_Complex16 *x_val,
                                            const ALPHA_INT *x_ind,
                                            ALPHA_Complex16 *y,
                                            alphasparse_index_base_t idx_base);





alphasparse_status_t alphasparse_dcu_s_bsrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT mb,
                                             ALPHA_INT nb,
                                             ALPHA_INT nnzb,
                                             const float *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const float *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT bsr_dim,
                                             const float *x,
                                             const float *beta,
                                             float *y);

alphasparse_status_t alphasparse_dcu_d_bsrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT mb,
                                             ALPHA_INT nb,
                                             ALPHA_INT nnzb,
                                             const double *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const double *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT bsr_dim,
                                             const double *x,
                                             const double *beta,
                                             double *y);

alphasparse_status_t alphasparse_dcu_c_bsrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT mb,
                                             ALPHA_INT nb,
                                             ALPHA_INT nnzb,
                                             const ALPHA_Complex8 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex8 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT bsr_dim,
                                             const ALPHA_Complex8 *x,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_dcu_z_bsrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT mb,
                                             ALPHA_INT nb,
                                             ALPHA_INT nnzb,
                                             const ALPHA_Complex16 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex16 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT bsr_dim,
                                             const ALPHA_Complex16 *x,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *y);



alphasparse_status_t alphasparse_dcu_bsrsv_zero_pivot(alphasparse_dcu_handle_t handle,
                                                      alphasparse_dcu_mat_info_t info,
                                                      ALPHA_INT *position);



alphasparse_status_t alphasparse_dcu_s_bsrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const float *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT bsr_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_d_bsrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const double *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT bsr_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_c_bsrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex8 *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT bsr_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_z_bsrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex16 *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT bsr_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_s_bsrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const float *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT bsr_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_d_bsrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const double *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT bsr_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_c_bsrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex8 *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT bsr_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_z_bsrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex16 *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT bsr_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);



alphasparse_status_t alphasparse_dcu_bsrsv_clear(alphasparse_dcu_handle_t handle, alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_s_bsrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_layout_t dir,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT mb,
                                                   ALPHA_INT nnzb,
                                                   const float *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const float *bsr_val,
                                                   const ALPHA_INT *bsr_row_ptr,
                                                   const ALPHA_INT *bsr_col_ind,
                                                   ALPHA_INT bsr_dim,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const float *x,
                                                   float *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_d_bsrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_layout_t dir,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT mb,
                                                   ALPHA_INT nnzb,
                                                   const double *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const double *bsr_val,
                                                   const ALPHA_INT *bsr_row_ptr,
                                                   const ALPHA_INT *bsr_col_ind,
                                                   ALPHA_INT bsr_dim,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const double *x,
                                                   double *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_c_bsrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_layout_t dir,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT mb,
                                                   ALPHA_INT nnzb,
                                                   const ALPHA_Complex8 *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const ALPHA_Complex8 *bsr_val,
                                                   const ALPHA_INT *bsr_row_ptr,
                                                   const ALPHA_INT *bsr_col_ind,
                                                   ALPHA_INT bsr_dim,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const ALPHA_Complex8 *x,
                                                   ALPHA_Complex8 *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_z_bsrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_layout_t dir,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT mb,
                                                   ALPHA_INT nnzb,
                                                   const ALPHA_Complex16 *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const ALPHA_Complex16 *bsr_val,
                                                   const ALPHA_INT *bsr_row_ptr,
                                                   const ALPHA_INT *bsr_col_ind,
                                                   ALPHA_INT bsr_dim,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const ALPHA_Complex16 *x,
                                                   ALPHA_Complex16 *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);



alphasparse_status_t alphasparse_dcu_s_coomv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const float *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const float *coo_val,
                                             const ALPHA_INT *coo_row_ind,
                                             const ALPHA_INT *coo_col_ind,
                                             const float *x,
                                             const float *beta,
                                             float *y);

alphasparse_status_t alphasparse_dcu_d_coomv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const double *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const double *coo_val,
                                             const ALPHA_INT *coo_row_ind,
                                             const ALPHA_INT *coo_col_ind,
                                             const double *x,
                                             const double *beta,
                                             double *y);

alphasparse_status_t alphasparse_dcu_c_coomv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex8 *coo_val,
                                             const ALPHA_INT *coo_row_ind,
                                             const ALPHA_INT *coo_col_ind,
                                             const ALPHA_Complex8 *x,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_dcu_z_coomv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex16 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex16 *coo_val,
                                             const ALPHA_INT *coo_row_ind,
                                             const ALPHA_INT *coo_col_ind,
                                             const ALPHA_Complex16 *x,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *y);



alphasparse_status_t alphasparse_dcu_s_csrmv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT n,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const float *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info);

alphasparse_status_t alphasparse_dcu_d_csrmv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT n,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const double *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info);

alphasparse_status_t alphasparse_dcu_c_csrmv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT n,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex8 *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info);

alphasparse_status_t alphasparse_dcu_z_csrmv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT n,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex16 *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_csrmv_clear(alphasparse_dcu_handle_t handle, alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_s_csrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const float *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const float *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const float *x,
                                             const float *beta,
                                             float *y);

alphasparse_status_t alphasparse_dcu_d_csrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const double *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const double *x,
                                             const double *beta,
                                             double *y);

alphasparse_status_t alphasparse_dcu_c_csrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const ALPHA_Complex8 *x,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_dcu_z_csrmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex16 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex16 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const ALPHA_Complex16 *x,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *y);

/**
 * csr5mv
 * 
 */
alphasparse_status_t alphasparse_dcu_s_csr5mv(alphasparse_dcu_handle_t handle,
                                              alphasparse_operation_t trans,
                                              const float *alpha,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const spmat_csr5_s_t *csr5,
                                              alphasparse_dcu_mat_info_t info,
                                              const float *x,
                                              const float *beta,
                                              float *y);

alphasparse_status_t alphasparse_dcu_d_csr5mv(alphasparse_dcu_handle_t handle,
                                              alphasparse_operation_t trans,
                                              const double *alpha,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const spmat_csr5_d_t *csr5,
                                              alphasparse_dcu_mat_info_t info,
                                              const double *x,
                                              const double *beta,
                                              double *y);

alphasparse_status_t alphasparse_dcu_c_csr5mv(alphasparse_dcu_handle_t handle,
                                              alphasparse_operation_t trans,
                                              const ALPHA_Complex8 *alpha,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const spmat_csr5_c_t *csr5,
                                              alphasparse_dcu_mat_info_t info,
                                              const ALPHA_Complex8 *x,
                                              const ALPHA_Complex8 *beta,
                                              ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_dcu_z_csr5mv(alphasparse_dcu_handle_t handle,
                                              alphasparse_operation_t trans,
                                              const ALPHA_Complex16 *alpha,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const spmat_csr5_z_t *csr5,
                                              alphasparse_dcu_mat_info_t info,
                                              const ALPHA_Complex16 *x,
                                              const ALPHA_Complex16 *beta,
                                              ALPHA_Complex16 *y);



alphasparse_status_t alphasparse_dcu_csrsv_zero_pivot(alphasparse_dcu_handle_t handle,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      alphasparse_dcu_mat_info_t info,
                                                      ALPHA_INT *position);



alphasparse_status_t alphasparse_dcu_s_csrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const float *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_d_csrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const double *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_c_csrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex8 *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_z_csrsv_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_operation_t trans,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex16 *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_s_csrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const float *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_d_csrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const double *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_c_csrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex8 *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_z_csrsv_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_operation_t trans,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex16 *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);



alphasparse_status_t alphasparse_dcu_csrsv_clear(alphasparse_dcu_handle_t handle,
                                                 const alpha_dcu_matrix_descr_t descr,
                                                 alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_s_csrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nnz,
                                                   const float *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const float *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const float *x,
                                                   float *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_d_csrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nnz,
                                                   const double *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const double *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const double *x,
                                                   double *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_c_csrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nnz,
                                                   const ALPHA_Complex8 *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const ALPHA_Complex8 *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const ALPHA_Complex8 *x,
                                                   ALPHA_Complex8 *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_z_csrsv_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nnz,
                                                   const ALPHA_Complex16 *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const ALPHA_Complex16 *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   alphasparse_dcu_mat_info_t info,
                                                   const ALPHA_Complex16 *x,
                                                   ALPHA_Complex16 *y,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);



alphasparse_status_t alphasparse_dcu_s_ellmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             const float *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const float *ell_val,
                                             const ALPHA_INT *ell_col_ind,
                                             ALPHA_INT ell_width,
                                             const float *x,
                                             const float *beta,
                                             float *y);

alphasparse_status_t alphasparse_dcu_d_ellmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             const double *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const double *ell_val,
                                             const ALPHA_INT *ell_col_ind,
                                             ALPHA_INT ell_width,
                                             const double *x,
                                             const double *beta,
                                             double *y);

alphasparse_status_t alphasparse_dcu_c_ellmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             const ALPHA_Complex8 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex8 *ell_val,
                                             const ALPHA_INT *ell_col_ind,
                                             ALPHA_INT ell_width,
                                             const ALPHA_Complex8 *x,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_dcu_z_ellmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             const ALPHA_Complex16 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex16 *ell_val,
                                             const ALPHA_INT *ell_col_ind,
                                             ALPHA_INT ell_width,
                                             const ALPHA_Complex16 *x,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *y);



alphasparse_status_t alphasparse_dcu_s_hybmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             const float *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const spmat_hyb_s_t *hyb,
                                             const float *x,
                                             const float *beta,
                                             float *y);

alphasparse_status_t alphasparse_dcu_d_hybmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             const double *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const spmat_hyb_d_t *hyb,
                                             const double *x,
                                             const double *beta,
                                             double *y);

alphasparse_status_t alphasparse_dcu_c_hybmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             const ALPHA_Complex8 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const spmat_hyb_c_t *hyb,
                                             const ALPHA_Complex8 *x,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_dcu_z_hybmv(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans,
                                             const ALPHA_Complex16 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const spmat_hyb_z_t *hyb,
                                             const ALPHA_Complex16 *x,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *y);



alphasparse_status_t alphasparse_dcu_s_gebsrmv(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans,
                                               ALPHA_INT mb,
                                               ALPHA_INT nb,
                                               ALPHA_INT nnzb,
                                               const float *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const float *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT col_block_dim,
                                               const float *x,
                                               const float *beta,
                                               float *y);

alphasparse_status_t alphasparse_dcu_d_gebsrmv(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans,
                                               ALPHA_INT mb,
                                               ALPHA_INT nb,
                                               ALPHA_INT nnzb,
                                               const double *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const double *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT col_block_dim,
                                               const double *x,
                                               const double *beta,
                                               double *y);

alphasparse_status_t alphasparse_dcu_c_gebsrmv(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans,
                                               ALPHA_INT mb,
                                               ALPHA_INT nb,
                                               ALPHA_INT nnzb,
                                               const ALPHA_Complex8 *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const ALPHA_Complex8 *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT col_block_dim,
                                               const ALPHA_Complex8 *x,
                                               const ALPHA_Complex8 *beta,
                                               ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_dcu_z_gebsrmv(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans,
                                               ALPHA_INT mb,
                                               ALPHA_INT nb,
                                               ALPHA_INT nnzb,
                                               const ALPHA_Complex16 *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const ALPHA_Complex16 *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT col_block_dim,
                                               const ALPHA_Complex16 *x,
                                               const ALPHA_Complex16 *beta,
                                               ALPHA_Complex16 *y);





alphasparse_status_t alphasparse_dcu_s_bsrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT mb,
                                             ALPHA_INT n,
                                             ALPHA_INT kb,
                                             ALPHA_INT nnzb,
                                             const float *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const float *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             const float *B,
                                             ALPHA_INT ldb,
                                             const float *beta,
                                             float *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_d_bsrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT mb,
                                             ALPHA_INT n,
                                             ALPHA_INT kb,
                                             ALPHA_INT nnzb,
                                             const double *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const double *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             const double *B,
                                             ALPHA_INT ldb,
                                             const double *beta,
                                             double *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_c_bsrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT mb,
                                             ALPHA_INT n,
                                             ALPHA_INT kb,
                                             ALPHA_INT nnzb,
                                             const ALPHA_Complex8 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex8 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             const ALPHA_Complex8 *B,
                                             ALPHA_INT ldb,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_z_bsrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT mb,
                                             ALPHA_INT n,
                                             ALPHA_INT kb,
                                             ALPHA_INT nnzb,
                                             const ALPHA_Complex16 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex16 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             const ALPHA_Complex16 *B,
                                             ALPHA_INT ldb,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *C,
                                             ALPHA_INT ldc);



alphasparse_status_t alphasparse_dcu_s_gebsrmm(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT mb,
                                               ALPHA_INT n,
                                               ALPHA_INT kb,
                                               ALPHA_INT nnzb,
                                               const float *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const float *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT col_block_dim,
                                               const float *B,
                                               ALPHA_INT ldb,
                                               const float *beta,
                                               float *C,
                                               ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_d_gebsrmm(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT mb,
                                               ALPHA_INT n,
                                               ALPHA_INT kb,
                                               ALPHA_INT nnzb,
                                               const double *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const double *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT col_block_dim,
                                               const double *B,
                                               ALPHA_INT ldb,
                                               const double *beta,
                                               double *C,
                                               ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_c_gebsrmm(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT mb,
                                               ALPHA_INT n,
                                               ALPHA_INT kb,
                                               ALPHA_INT nnzb,
                                               const ALPHA_Complex8 *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const ALPHA_Complex8 *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT col_block_dim,
                                               const ALPHA_Complex8 *B,
                                               ALPHA_INT ldb,
                                               const ALPHA_Complex8 *beta,
                                               ALPHA_Complex8 *C,
                                               ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_z_gebsrmm(alphasparse_dcu_handle_t handle,
                                               alphasparse_layout_t dir,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT mb,
                                               ALPHA_INT n,
                                               ALPHA_INT kb,
                                               ALPHA_INT nnzb,
                                               const ALPHA_Complex16 *alpha,
                                               const alpha_dcu_matrix_descr_t descr,
                                               const ALPHA_Complex16 *bsr_val,
                                               const ALPHA_INT *bsr_row_ptr,
                                               const ALPHA_INT *bsr_col_ind,
                                               ALPHA_INT row_block_dim,
                                               ALPHA_INT colblock_dim,
                                               const ALPHA_Complex16 *B,
                                               ALPHA_INT ldb,
                                               const ALPHA_Complex16 *beta,
                                               ALPHA_Complex16 *C,
                                               ALPHA_INT ldc);



alphasparse_status_t alphasparse_dcu_s_csrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             alphasparse_layout_t layout,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const float *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const float *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const float *B,
                                             ALPHA_INT ldb,
                                             const float *beta,
                                             float *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_d_csrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             alphasparse_layout_t layout,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const double *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const double *B,
                                             ALPHA_INT ldb,
                                             const double *beta,
                                             double *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_c_csrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             alphasparse_layout_t layout,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const ALPHA_Complex8 *B,
                                             ALPHA_INT ldb,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_z_csrmm(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             alphasparse_layout_t layout,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex16 *alpha,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex16 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const ALPHA_Complex16 *B,
                                             ALPHA_INT ldb,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *C,
                                             ALPHA_INT ldc);



alphasparse_status_t alphasparse_dcu_csrsm_zero_pivot(alphasparse_dcu_handle_t handle,
                                                      alphasparse_dcu_mat_info_t info,
                                                      ALPHA_INT *position);



alphasparse_status_t alphasparse_dcu_scsrsm_buffer_size(alphasparse_dcu_handle_t handle,
                                                        alphasparse_operation_t trans_A,
                                                        alphasparse_operation_t trans_B,
                                                        ALPHA_INT m,
                                                        ALPHA_INT nrhs,
                                                        ALPHA_INT nnz,
                                                        const float *alpha,
                                                        const alpha_dcu_matrix_descr_t descr,
                                                        const float *csr_val,
                                                        const ALPHA_INT *csr_row_ptr,
                                                        const ALPHA_INT *csr_col_ind,
                                                        const float *B,
                                                        ALPHA_INT ldb,
                                                        alphasparse_dcu_mat_info_t info,
                                                        alphasparse_dcu_solve_policy_t policy,
                                                        size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dcsrsm_buffer_size(alphasparse_dcu_handle_t handle,
                                                        alphasparse_operation_t trans_A,
                                                        alphasparse_operation_t trans_B,
                                                        ALPHA_INT m,
                                                        ALPHA_INT nrhs,
                                                        ALPHA_INT nnz,
                                                        const double *alpha,
                                                        const alpha_dcu_matrix_descr_t descr,
                                                        const double *csr_val,
                                                        const ALPHA_INT *csr_row_ptr,
                                                        const ALPHA_INT *csr_col_ind,
                                                        const double *B,
                                                        ALPHA_INT ldb,
                                                        alphasparse_dcu_mat_info_t info,
                                                        alphasparse_dcu_solve_policy_t policy,
                                                        size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_ccsrsm_buffer_size(alphasparse_dcu_handle_t handle,
                                                        alphasparse_operation_t trans_A,
                                                        alphasparse_operation_t trans_B,
                                                        ALPHA_INT m,
                                                        ALPHA_INT nrhs,
                                                        ALPHA_INT nnz,
                                                        const ALPHA_Complex8 *alpha,
                                                        const alpha_dcu_matrix_descr_t descr,
                                                        const ALPHA_Complex8 *csr_val,
                                                        const ALPHA_INT *csr_row_ptr,
                                                        const ALPHA_INT *csr_col_ind,
                                                        const ALPHA_Complex8 *B,
                                                        ALPHA_INT ldb,
                                                        alphasparse_dcu_mat_info_t info,
                                                        alphasparse_dcu_solve_policy_t policy,
                                                        size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_zcsrsm_buffer_size(alphasparse_dcu_handle_t handle,
                                                        alphasparse_operation_t trans_A,
                                                        alphasparse_operation_t trans_B,
                                                        ALPHA_INT m,
                                                        ALPHA_INT nrhs,
                                                        ALPHA_INT nnz,
                                                        const ALPHA_Complex16 *alpha,
                                                        const alpha_dcu_matrix_descr_t descr,
                                                        const ALPHA_Complex16 *csr_val,
                                                        const ALPHA_INT *csr_row_ptr,
                                                        const ALPHA_INT *csr_col_ind,
                                                        const ALPHA_Complex16 *B,
                                                        ALPHA_INT ldb,
                                                        alphasparse_dcu_mat_info_t info,
                                                        alphasparse_dcu_solve_policy_t policy,
                                                        size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_scsrsm_analysis(alphasparse_dcu_handle_t handle,
                                                     alphasparse_operation_t trans_A,
                                                     alphasparse_operation_t trans_B,
                                                     ALPHA_INT m,
                                                     ALPHA_INT nrhs,
                                                     ALPHA_INT nnz,
                                                     const float *alpha,
                                                     const alpha_dcu_matrix_descr_t descr,
                                                     const float *csr_val,
                                                     const ALPHA_INT *csr_row_ptr,
                                                     const ALPHA_INT *csr_col_ind,
                                                     const float *B,
                                                     ALPHA_INT ldb,
                                                     alphasparse_dcu_mat_info_t info,
                                                     alphasparse_dcu_analysis_policy_t analysis,
                                                     alphasparse_dcu_solve_policy_t solve,
                                                     void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dcsrsm_analysis(alphasparse_dcu_handle_t handle,
                                                     alphasparse_operation_t trans_A,
                                                     alphasparse_operation_t trans_B,
                                                     ALPHA_INT m,
                                                     ALPHA_INT nrhs,
                                                     ALPHA_INT nnz,
                                                     const double *alpha,
                                                     const alpha_dcu_matrix_descr_t descr,
                                                     const double *csr_val,
                                                     const ALPHA_INT *csr_row_ptr,
                                                     const ALPHA_INT *csr_col_ind,
                                                     const double *B,
                                                     ALPHA_INT ldb,
                                                     alphasparse_dcu_mat_info_t info,
                                                     alphasparse_dcu_analysis_policy_t analysis,
                                                     alphasparse_dcu_solve_policy_t solve,
                                                     void *temp_buffer);

alphasparse_status_t alphasparse_dcu_ccsrsm_analysis(alphasparse_dcu_handle_t handle,
                                                     alphasparse_operation_t trans_A,
                                                     alphasparse_operation_t trans_B,
                                                     ALPHA_INT m,
                                                     ALPHA_INT nrhs,
                                                     ALPHA_INT nnz,
                                                     const ALPHA_Complex8 *alpha,
                                                     const alpha_dcu_matrix_descr_t descr,
                                                     const ALPHA_Complex8 *csr_val,
                                                     const ALPHA_INT *csr_row_ptr,
                                                     const ALPHA_INT *csr_col_ind,
                                                     const ALPHA_Complex8 *B,
                                                     ALPHA_INT ldb,
                                                     alphasparse_dcu_mat_info_t info,
                                                     alphasparse_dcu_analysis_policy_t analysis,
                                                     alphasparse_dcu_solve_policy_t solve,
                                                     void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zcsrsm_analysis(alphasparse_dcu_handle_t handle,
                                                     alphasparse_operation_t trans_A,
                                                     alphasparse_operation_t trans_B,
                                                     ALPHA_INT m,
                                                     ALPHA_INT nrhs,
                                                     ALPHA_INT nnz,
                                                     const ALPHA_Complex16 *alpha,
                                                     const alpha_dcu_matrix_descr_t descr,
                                                     const ALPHA_Complex16 *csr_val,
                                                     const ALPHA_INT *csr_row_ptr,
                                                     const ALPHA_INT *csr_col_ind,
                                                     const ALPHA_Complex16 *B,
                                                     ALPHA_INT ldb,
                                                     alphasparse_dcu_mat_info_t info,
                                                     alphasparse_dcu_analysis_policy_t analysis,
                                                     alphasparse_dcu_solve_policy_t solve,
                                                     void *temp_buffer);



alphasparse_status_t alphasparse_dcu_csrsm_clear(alphasparse_dcu_handle_t handle, alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_s_csrsm_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans_A,
                                                   alphasparse_operation_t trans_B,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nrhs,
                                                   ALPHA_INT nnz,
                                                   const float *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const float *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   float *B,
                                                   ALPHA_INT ldb,
                                                   alphasparse_dcu_mat_info_t info,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_d_csrsm_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans_A,
                                                   alphasparse_operation_t trans_B,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nrhs,
                                                   ALPHA_INT nnz,
                                                   const double *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const double *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   double *B,
                                                   ALPHA_INT ldb,
                                                   alphasparse_dcu_mat_info_t info,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_c_csrsm_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans_A,
                                                   alphasparse_operation_t trans_B,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nrhs,
                                                   ALPHA_INT nnz,
                                                   const ALPHA_Complex8 *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const ALPHA_Complex8 *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   ALPHA_Complex8 *B,
                                                   ALPHA_INT ldb,
                                                   alphasparse_dcu_mat_info_t info,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);

alphasparse_status_t alphasparse_dcu_z_csrsm_solve(alphasparse_dcu_handle_t handle,
                                                   alphasparse_operation_t trans_A,
                                                   alphasparse_operation_t trans_B,
                                                   ALPHA_INT m,
                                                   ALPHA_INT nrhs,
                                                   ALPHA_INT nnz,
                                                   const ALPHA_Complex16 *alpha,
                                                   const alpha_dcu_matrix_descr_t descr,
                                                   const ALPHA_Complex16 *csr_val,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   ALPHA_Complex16 *B,
                                                   ALPHA_INT ldb,
                                                   alphasparse_dcu_mat_info_t info,
                                                   alphasparse_dcu_solve_policy_t policy,
                                                   void *temp_buffer);



alphasparse_status_t alphasparse_dcu_s_gemmi(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const float *alpha,
                                             const float *A,
                                             ALPHA_INT lda,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const float *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const float *beta,
                                             float *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_d_gemmi(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const double *alpha,
                                             const double *A,
                                             ALPHA_INT lda,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const double *beta,
                                             double *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_c_gemmi(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex8 *alpha,
                                             const ALPHA_Complex8 *A,
                                             ALPHA_INT lda,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const ALPHA_Complex8 *beta,
                                             ALPHA_Complex8 *C,
                                             ALPHA_INT ldc);

alphasparse_status_t alphasparse_dcu_z_gemmi(alphasparse_dcu_handle_t handle,
                                             alphasparse_operation_t trans_A,
                                             alphasparse_operation_t trans_B,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             ALPHA_INT nnz,
                                             const ALPHA_Complex16 *alpha,
                                             const ALPHA_Complex16 *A,
                                             ALPHA_INT lda,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_Complex16 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             const ALPHA_Complex16 *beta,
                                             ALPHA_Complex16 *C,
                                             ALPHA_INT ldc);





alphasparse_status_t alphasparse_dcu_csrgeam_nnz(alphasparse_dcu_handle_t handle,
                                                 ALPHA_INT m,
                                                 ALPHA_INT n,
                                                 const alpha_dcu_matrix_descr_t descr_A,
                                                 ALPHA_INT nnz_A,
                                                 const ALPHA_INT *csr_row_ptr_A,
                                                 const ALPHA_INT *csr_col_ind_A,
                                                 const alpha_dcu_matrix_descr_t descr_B,
                                                 ALPHA_INT nnz_B,
                                                 const ALPHA_INT *csr_row_ptr_B,
                                                 const ALPHA_INT *csr_col_ind_B,
                                                 const alpha_dcu_matrix_descr_t descr_C,
                                                 ALPHA_INT *csr_row_ptr_C,
                                                 ALPHA_INT *nnz_C);



alphasparse_status_t alphasparse_dcu_s_csrgeam(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               const float *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const float *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const float *beta,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const float *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               float *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C);

alphasparse_status_t alphasparse_dcu_d_csrgeam(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               const double *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const double *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const double *beta,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const double *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               double *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C);

alphasparse_status_t alphasparse_dcu_c_csrgeam(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               const ALPHA_Complex8 *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const ALPHA_Complex8 *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const ALPHA_Complex8 *beta,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const ALPHA_Complex8 *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               ALPHA_Complex8 *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C);

alphasparse_status_t alphasparse_dcu_z_csrgeam(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               const ALPHA_Complex16 *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const ALPHA_Complex16 *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const ALPHA_Complex16 *beta,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const ALPHA_Complex16 *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               ALPHA_Complex16 *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C);



alphasparse_status_t alphasparse_dcu_scsrgemm_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_operation_t trans_A,
                                                          alphasparse_operation_t trans_B,
                                                          ALPHA_INT m,
                                                          ALPHA_INT n,
                                                          ALPHA_INT k,
                                                          const float *alpha,
                                                          const alpha_dcu_matrix_descr_t descr_A,
                                                          ALPHA_INT nnz_A,
                                                          const ALPHA_INT *csr_row_ptr_A,
                                                          const ALPHA_INT *csr_col_ind_A,
                                                          const alpha_dcu_matrix_descr_t descr_B,
                                                          ALPHA_INT nnz_B,
                                                          const ALPHA_INT *csr_row_ptr_B,
                                                          const ALPHA_INT *csr_col_ind_B,
                                                          const float *beta,
                                                          const alpha_dcu_matrix_descr_t descr_D,
                                                          ALPHA_INT nnz_D,
                                                          const ALPHA_INT *csr_row_ptr_D,
                                                          const ALPHA_INT *csr_col_ind_D,
                                                          alphasparse_dcu_mat_info_t info_C,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dcsrgemm_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_operation_t trans_A,
                                                          alphasparse_operation_t trans_B,
                                                          ALPHA_INT m,
                                                          ALPHA_INT n,
                                                          ALPHA_INT k,
                                                          const double *alpha,
                                                          const alpha_dcu_matrix_descr_t descr_A,
                                                          ALPHA_INT nnz_A,
                                                          const ALPHA_INT *csr_row_ptr_A,
                                                          const ALPHA_INT *csr_col_ind_A,
                                                          const alpha_dcu_matrix_descr_t descr_B,
                                                          ALPHA_INT nnz_B,
                                                          const ALPHA_INT *csr_row_ptr_B,
                                                          const ALPHA_INT *csr_col_ind_B,
                                                          const double *beta,
                                                          const alpha_dcu_matrix_descr_t descr_D,
                                                          ALPHA_INT nnz_D,
                                                          const ALPHA_INT *csr_row_ptr_D,
                                                          const ALPHA_INT *csr_col_ind_D,
                                                          alphasparse_dcu_mat_info_t info_C,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_ccsrgemm_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_operation_t trans_A,
                                                          alphasparse_operation_t trans_B,
                                                          ALPHA_INT m,
                                                          ALPHA_INT n,
                                                          ALPHA_INT k,
                                                          const ALPHA_Complex8 *alpha,
                                                          const alpha_dcu_matrix_descr_t descr_A,
                                                          ALPHA_INT nnz_A,
                                                          const ALPHA_INT *csr_row_ptr_A,
                                                          const ALPHA_INT *csr_col_ind_A,
                                                          const alpha_dcu_matrix_descr_t descr_B,
                                                          ALPHA_INT nnz_B,
                                                          const ALPHA_INT *csr_row_ptr_B,
                                                          const ALPHA_INT *csr_col_ind_B,
                                                          const ALPHA_Complex8 *beta,
                                                          const alpha_dcu_matrix_descr_t descr_D,
                                                          ALPHA_INT nnz_D,
                                                          const ALPHA_INT *csr_row_ptr_D,
                                                          const ALPHA_INT *csr_col_ind_D,
                                                          alphasparse_dcu_mat_info_t info_C,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_zcsrgemm_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_operation_t trans_A,
                                                          alphasparse_operation_t trans_B,
                                                          ALPHA_INT m,
                                                          ALPHA_INT n,
                                                          ALPHA_INT k,
                                                          const ALPHA_Complex16 *alpha,
                                                          const alpha_dcu_matrix_descr_t descr_A,
                                                          ALPHA_INT nnz_A,
                                                          const ALPHA_INT *csr_row_ptr_A,
                                                          const ALPHA_INT *csr_col_ind_A,
                                                          const alpha_dcu_matrix_descr_t descr_B,
                                                          ALPHA_INT nnz_B,
                                                          const ALPHA_INT *csr_row_ptr_B,
                                                          const ALPHA_INT *csr_col_ind_B,
                                                          const ALPHA_Complex16 *beta,
                                                          const alpha_dcu_matrix_descr_t descr_D,
                                                          ALPHA_INT nnz_D,
                                                          const ALPHA_INT *csr_row_ptr_D,
                                                          const ALPHA_INT *csr_col_ind_D,
                                                          alphasparse_dcu_mat_info_t info_C,
                                                          size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_csrgemm_nnz(alphasparse_dcu_handle_t handle,
                                                 alphasparse_operation_t trans_A,
                                                 alphasparse_operation_t trans_B,
                                                 ALPHA_INT m,
                                                 ALPHA_INT n,
                                                 ALPHA_INT k,
                                                 const alpha_dcu_matrix_descr_t descr_A,
                                                 ALPHA_INT nnz_A,
                                                 const ALPHA_INT *csr_row_ptr_A,
                                                 const ALPHA_INT *csr_col_ind_A,
                                                 const alpha_dcu_matrix_descr_t descr_B,
                                                 ALPHA_INT nnz_B,
                                                 const ALPHA_INT *csr_row_ptr_B,
                                                 const ALPHA_INT *csr_col_ind_B,
                                                 const alpha_dcu_matrix_descr_t descr_D,
                                                 ALPHA_INT nnz_D,
                                                 const ALPHA_INT *csr_row_ptr_D,
                                                 const ALPHA_INT *csr_col_ind_D,
                                                 const alpha_dcu_matrix_descr_t descr_C,
                                                 ALPHA_INT *csr_row_ptr_C,
                                                 ALPHA_INT *nnz_C,
                                                 const alphasparse_dcu_mat_info_t info_C,
                                                 void *temp_buffer);



alphasparse_status_t alphasparse_dcu_s_csrgemm(alphasparse_dcu_handle_t handle,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               const float *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const float *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const float *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const float *beta,
                                               const alpha_dcu_matrix_descr_t descr_D,
                                               ALPHA_INT nnz_D,
                                               const float *csr_val_D,
                                               const ALPHA_INT *csr_row_ptr_D,
                                               const ALPHA_INT *csr_col_ind_D,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               float *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C,
                                               const alphasparse_dcu_mat_info_t info_C,
                                               void *temp_buffer);

alphasparse_status_t alphasparse_dcu_d_csrgemm(alphasparse_dcu_handle_t handle,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               const double *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const double *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const double *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const double *beta,
                                               const alpha_dcu_matrix_descr_t descr_D,
                                               ALPHA_INT nnz_D,
                                               const double *csr_val_D,
                                               const ALPHA_INT *csr_row_ptr_D,
                                               const ALPHA_INT *csr_col_ind_D,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               double *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C,
                                               const alphasparse_dcu_mat_info_t info_C,
                                               void *temp_buffer);

alphasparse_status_t alphasparse_dcu_c_csrgemm(alphasparse_dcu_handle_t handle,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               const ALPHA_Complex8 *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const ALPHA_Complex8 *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const ALPHA_Complex8 *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const ALPHA_Complex8 *beta,
                                               const alpha_dcu_matrix_descr_t descr_D,
                                               ALPHA_INT nnz_D,
                                               const ALPHA_Complex8 *csr_val_D,
                                               const ALPHA_INT *csr_row_ptr_D,
                                               const ALPHA_INT *csr_col_ind_D,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               ALPHA_Complex8 *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C,
                                               const alphasparse_dcu_mat_info_t info_C,
                                               void *temp_buffer);

alphasparse_status_t alphasparse_dcu_z_csrgemm(alphasparse_dcu_handle_t handle,
                                               alphasparse_operation_t trans_A,
                                               alphasparse_operation_t trans_B,
                                               ALPHA_INT m,
                                               ALPHA_INT n,
                                               ALPHA_INT k,
                                               const ALPHA_Complex16 *alpha,
                                               const alpha_dcu_matrix_descr_t descr_A,
                                               ALPHA_INT nnz_A,
                                               const ALPHA_Complex16 *csr_val_A,
                                               const ALPHA_INT *csr_row_ptr_A,
                                               const ALPHA_INT *csr_col_ind_A,
                                               const alpha_dcu_matrix_descr_t descr_B,
                                               ALPHA_INT nnz_B,
                                               const ALPHA_Complex16 *csr_val_B,
                                               const ALPHA_INT *csr_row_ptr_B,
                                               const ALPHA_INT *csr_col_ind_B,
                                               const ALPHA_Complex16 *beta,
                                               const alpha_dcu_matrix_descr_t descr_D,
                                               ALPHA_INT nnz_D,
                                               const ALPHA_Complex16 *csr_val_D,
                                               const ALPHA_INT *csr_row_ptr_D,
                                               const ALPHA_INT *csr_col_ind_D,
                                               const alpha_dcu_matrix_descr_t descr_C,
                                               ALPHA_Complex16 *csr_val_C,
                                               const ALPHA_INT *csr_row_ptr_C,
                                               ALPHA_INT *csr_col_ind_C,
                                               const alphasparse_dcu_mat_info_t info_C,
                                               void *temp_buffer);





alphasparse_status_t alphasparse_dcu_bsric0_zero_pivot(alphasparse_dcu_handle_t handle,
                                                       alphasparse_dcu_mat_info_t info,
                                                       ALPHA_INT *position);



alphasparse_status_t alphasparse_dcu_sbsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const float *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT block_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dbsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const double *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT block_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_cbsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex8 *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT block_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_zbsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         alphasparse_layout_t dir,
                                                         ALPHA_INT mb,
                                                         ALPHA_INT nnzb,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex16 *bsr_val,
                                                         const ALPHA_INT *bsr_row_ptr,
                                                         const ALPHA_INT *bsr_col_ind,
                                                         ALPHA_INT block_dim,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_sbsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const float *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT block_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dbsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const double *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT block_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_cbsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex8 *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT block_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zbsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      alphasparse_layout_t dir,
                                                      ALPHA_INT mb,
                                                      ALPHA_INT nnzb,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex16 *bsr_val,
                                                      const ALPHA_INT *bsr_row_ptr,
                                                      const ALPHA_INT *bsr_col_ind,
                                                      ALPHA_INT block_dim,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);



alphasparse_status_t alphasparse_dcu_bsric0_clear(alphasparse_dcu_handle_t handle, alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_sbsric0(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             ALPHA_INT mb,
                                             ALPHA_INT nnzb,
                                             const alpha_dcu_matrix_descr_t descr,
                                             float *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dbsric0(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             ALPHA_INT mb,
                                             ALPHA_INT nnzb,
                                             const alpha_dcu_matrix_descr_t descr,
                                             double *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

alphasparse_status_t alphasparse_dcu_cbsric0(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             ALPHA_INT mb,
                                             ALPHA_INT nnzb,
                                             const alpha_dcu_matrix_descr_t descr,
                                             ALPHA_Complex8 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zbsric0(alphasparse_dcu_handle_t handle,
                                             alphasparse_layout_t dir,
                                             ALPHA_INT mb,
                                             ALPHA_INT nnzb,
                                             const alpha_dcu_matrix_descr_t descr,
                                             ALPHA_Complex16 *bsr_val,
                                             const ALPHA_INT *bsr_row_ptr,
                                             const ALPHA_INT *bsr_col_ind,
                                             ALPHA_INT block_dim,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);



alphasparse_status_t alphasparse_dcu_bsrilu0_zero_pivot(alphasparse_dcu_handle_t handle,
                                                        alphasparse_dcu_mat_info_t info,
                                                        ALPHA_INT *position);



alphasparse_status_t alphasparse_dcu_sbsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const float *boost_tol,
                                                            const float *boost_val);

alphasparse_status_t alphasparse_dcu_dbsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const double *boost_tol,
                                                            const double *boost_val);

alphasparse_status_t alphasparse_dcu_cbsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const float *boost_tol,
                                                            const ALPHA_Complex8 *boost_val);

alphasparse_status_t alphasparse_dcu_zbsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const double *boost_tol,
                                                            const ALPHA_Complex16 *boost_val);



alphasparse_status_t alphasparse_dcu_sbsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_layout_t dir,
                                                          ALPHA_INT mb,
                                                          ALPHA_INT nnzb,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const float *bsr_val,
                                                          const ALPHA_INT *bsr_row_ptr,
                                                          const ALPHA_INT *bsr_col_ind,
                                                          ALPHA_INT block_dim,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dbsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_layout_t dir,
                                                          ALPHA_INT mb,
                                                          ALPHA_INT nnzb,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const double *bsr_val,
                                                          const ALPHA_INT *bsr_row_ptr,
                                                          const ALPHA_INT *bsr_col_ind,
                                                          ALPHA_INT block_dim,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_cbsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_layout_t dir,
                                                          ALPHA_INT mb,
                                                          ALPHA_INT nnzb,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const ALPHA_Complex8 *bsr_val,
                                                          const ALPHA_INT *bsr_row_ptr,
                                                          const ALPHA_INT *bsr_col_ind,
                                                          ALPHA_INT block_dim,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_zbsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          alphasparse_layout_t dir,
                                                          ALPHA_INT mb,
                                                          ALPHA_INT nnzb,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const ALPHA_Complex16 *bsr_val,
                                                          const ALPHA_INT *bsr_row_ptr,
                                                          const ALPHA_INT *bsr_col_ind,
                                                          ALPHA_INT block_dim,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_sbsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       alphasparse_layout_t dir,
                                                       ALPHA_INT mb,
                                                       ALPHA_INT nnzb,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const float *bsr_val,
                                                       const ALPHA_INT *bsr_row_ptr,
                                                       const ALPHA_INT *bsr_col_ind,
                                                       ALPHA_INT block_dim,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dbsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       alphasparse_layout_t dir,
                                                       ALPHA_INT mb,
                                                       ALPHA_INT nnzb,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const double *bsr_val,
                                                       const ALPHA_INT *bsr_row_ptr,
                                                       const ALPHA_INT *bsr_col_ind,
                                                       ALPHA_INT block_dim,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);

alphasparse_status_t alphasparse_dcu_cbsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       alphasparse_layout_t dir,
                                                       ALPHA_INT mb,
                                                       ALPHA_INT nnzb,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const ALPHA_Complex8 *bsr_val,
                                                       const ALPHA_INT *bsr_row_ptr,
                                                       const ALPHA_INT *bsr_col_ind,
                                                       ALPHA_INT block_dim,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zbsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       alphasparse_layout_t dir,
                                                       ALPHA_INT mb,
                                                       ALPHA_INT nnzb,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const ALPHA_Complex16 *bsr_val,
                                                       const ALPHA_INT *bsr_row_ptr,
                                                       const ALPHA_INT *bsr_col_ind,
                                                       ALPHA_INT block_dim,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);



alphasparse_status_t alphasparse_dcu_bsrilu0_clear(alphasparse_dcu_handle_t handle, alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_sbsrilu0(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nnzb,
                                              const alpha_dcu_matrix_descr_t descr,
                                              float *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dbsrilu0(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nnzb,
                                              const alpha_dcu_matrix_descr_t descr,
                                              double *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_cbsrilu0(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nnzb,
                                              const alpha_dcu_matrix_descr_t descr,
                                              ALPHA_Complex8 *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zbsrilu0(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nnzb,
                                              const alpha_dcu_matrix_descr_t descr,
                                              ALPHA_Complex16 *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);



alphasparse_status_t alphasparse_dcu_csric0_zero_pivot(alphasparse_dcu_handle_t handle,
                                                       alphasparse_dcu_mat_info_t info,
                                                       ALPHA_INT *position);



alphasparse_status_t alphasparse_dcu_scsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const float *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dcsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const double *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_ccsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex8 *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_zcsric0_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT nnz,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const ALPHA_Complex16 *csr_val,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_scsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const float *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dcsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const double *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_ccsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex8 *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zcsric0_analysis(alphasparse_dcu_handle_t handle,
                                                      ALPHA_INT m,
                                                      ALPHA_INT nnz,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      const ALPHA_Complex16 *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      const ALPHA_INT *csr_col_ind,
                                                      alphasparse_dcu_mat_info_t info,
                                                      alphasparse_dcu_analysis_policy_t analysis,
                                                      alphasparse_dcu_solve_policy_t solve,
                                                      void *temp_buffer);



alphasparse_status_t alphasparse_dcu_csric0_clear(alphasparse_dcu_handle_t handle, alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_scsric0(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const alpha_dcu_matrix_descr_t descr,
                                             float *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dcsric0(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const alpha_dcu_matrix_descr_t descr,
                                             double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

alphasparse_status_t alphasparse_dcu_ccsric0(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const alpha_dcu_matrix_descr_t descr,
                                             ALPHA_Complex8 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zcsric0(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const alpha_dcu_matrix_descr_t descr,
                                             ALPHA_Complex16 *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);



alphasparse_status_t alphasparse_dcu_csrilu0_zero_pivot(alphasparse_dcu_handle_t handle,
                                                        alphasparse_dcu_mat_info_t info,
                                                        ALPHA_INT *position);



alphasparse_status_t alphasparse_dcu_scsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const float *boost_tol,
                                                            const float *boost_val);

alphasparse_status_t alphasparse_dcu_dcsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const double *boost_tol,
                                                            const double *boost_val);

alphasparse_status_t alphasparse_dcu_ccsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const float *boost_tol,
                                                            const ALPHA_Complex8 *boost_val);

alphasparse_status_t alphasparse_dcu_zcsrilu0_numeric_boost(alphasparse_dcu_handle_t handle,
                                                            alphasparse_dcu_mat_info_t info,
                                                            int enable_boost,
                                                            const double *boost_tol,
                                                            const ALPHA_Complex16 *boost_val);



alphasparse_status_t alphasparse_dcu_scsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          ALPHA_INT m,
                                                          ALPHA_INT nnz,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const float *csr_val,
                                                          const ALPHA_INT *csr_row_ptr,
                                                          const ALPHA_INT *csr_col_ind,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dcsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          ALPHA_INT m,
                                                          ALPHA_INT nnz,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const double *csr_val,
                                                          const ALPHA_INT *csr_row_ptr,
                                                          const ALPHA_INT *csr_col_ind,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_ccsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          ALPHA_INT m,
                                                          ALPHA_INT nnz,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const ALPHA_Complex8 *csr_val,
                                                          const ALPHA_INT *csr_row_ptr,
                                                          const ALPHA_INT *csr_col_ind,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_zcsrilu0_buffer_size(alphasparse_dcu_handle_t handle,
                                                          ALPHA_INT m,
                                                          ALPHA_INT nnz,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          const ALPHA_Complex16 *csr_val,
                                                          const ALPHA_INT *csr_row_ptr,
                                                          const ALPHA_INT *csr_col_ind,
                                                          alphasparse_dcu_mat_info_t info,
                                                          size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_scsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT nnz,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const float *csr_val,
                                                       const ALPHA_INT *csr_row_ptr,
                                                       const ALPHA_INT *csr_col_ind,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dcsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT nnz,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const double *csr_val,
                                                       const ALPHA_INT *csr_row_ptr,
                                                       const ALPHA_INT *csr_col_ind,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);

alphasparse_status_t alphasparse_dcu_ccsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT nnz,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const ALPHA_Complex8 *csr_val,
                                                       const ALPHA_INT *csr_row_ptr,
                                                       const ALPHA_INT *csr_col_ind,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zcsrilu0_analysis(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT nnz,
                                                       const alpha_dcu_matrix_descr_t descr,
                                                       const ALPHA_Complex16 *csr_val,
                                                       const ALPHA_INT *csr_row_ptr,
                                                       const ALPHA_INT *csr_col_ind,
                                                       alphasparse_dcu_mat_info_t info,
                                                       alphasparse_dcu_analysis_policy_t analysis,
                                                       alphasparse_dcu_solve_policy_t solve,
                                                       void *temp_buffer);



alphasparse_status_t alphasparse_dcu_csrilu0_clear(alphasparse_dcu_handle_t handle, alphasparse_dcu_mat_info_t info);



alphasparse_status_t alphasparse_dcu_scsrilu0(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const alpha_dcu_matrix_descr_t descr,
                                              float *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dcsrilu0(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const alpha_dcu_matrix_descr_t descr,
                                              double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_ccsrilu0(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const alpha_dcu_matrix_descr_t descr,
                                              ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zcsrilu0(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const alpha_dcu_matrix_descr_t descr,
                                              ALPHA_Complex16 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);





alphasparse_status_t alphasparse_dcu_snnz(alphasparse_dcu_handle_t handle,
                                          alphasparse_layout_t dir,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const float *A,
                                          ALPHA_INT ld,
                                          ALPHA_INT *nnz_per_row_columns,
                                          ALPHA_INT *nnz_total_dev_host_ptr);

alphasparse_status_t alphasparse_dcu_dnnz(alphasparse_dcu_handle_t handle,
                                          alphasparse_layout_t dir,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const double *A,
                                          ALPHA_INT ld,
                                          ALPHA_INT *nnz_per_row_columns,
                                          ALPHA_INT *nnz_total_dev_host_ptr);

alphasparse_status_t alphasparse_dcu_cnnz(alphasparse_dcu_handle_t handle,
                                          alphasparse_layout_t dir,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const ALPHA_Complex8 *A,
                                          ALPHA_INT ld,
                                          ALPHA_INT *nnz_per_row_columns,
                                          ALPHA_INT *nnz_total_dev_host_ptr);

alphasparse_status_t alphasparse_dcu_znnz(alphasparse_dcu_handle_t handle,
                                          alphasparse_layout_t dir,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const ALPHA_Complex16 *A,
                                          ALPHA_INT ld,
                                          ALPHA_INT *nnz_per_row_columns,
                                          ALPHA_INT *nnz_total_dev_host_ptr);



alphasparse_status_t alphasparse_dcu_sdense2csr(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const float *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                float *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_ddense2csr(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const double *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                double *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_cdense2csr(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex8 *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                ALPHA_Complex8 *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_zdense2csr(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex16 *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                ALPHA_Complex16 *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);



alphasparse_status_t alphasparse_dcu_sprune_dense2csr_buffer_size(alphasparse_dcu_handle_t handle,
                                                                  ALPHA_INT m,
                                                                  ALPHA_INT n,
                                                                  const float *A,
                                                                  ALPHA_INT lda,
                                                                  const float *threshold,
                                                                  const alpha_dcu_matrix_descr_t descr,
                                                                  const float *csr_val,
                                                                  const ALPHA_INT *csr_row_ptr,
                                                                  const ALPHA_INT *csr_col_ind,
                                                                  size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dprune_dense2csr_buffer_size(alphasparse_dcu_handle_t handle,
                                                                  ALPHA_INT m,
                                                                  ALPHA_INT n,
                                                                  const double *A,
                                                                  ALPHA_INT lda,
                                                                  const double *threshold,
                                                                  const alpha_dcu_matrix_descr_t descr,
                                                                  const double *csr_val,
                                                                  const ALPHA_INT *csr_row_ptr,
                                                                  const ALPHA_INT *csr_col_ind,
                                                                  size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_sprune_dense2csr_nnz(alphasparse_dcu_handle_t handle,
                                                          ALPHA_INT m,
                                                          ALPHA_INT n,
                                                          const float *A,
                                                          ALPHA_INT lda,
                                                          const float *threshold,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          ALPHA_INT *csr_row_ptr,
                                                          ALPHA_INT *nnz_total_dev_host_ptr,
                                                          void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_dense2csr_nnz(alphasparse_dcu_handle_t handle,
                                                          ALPHA_INT m,
                                                          ALPHA_INT n,
                                                          const double *A,
                                                          ALPHA_INT lda,
                                                          const double *threshold,
                                                          const alpha_dcu_matrix_descr_t descr,
                                                          ALPHA_INT *csr_row_ptr,
                                                          ALPHA_INT *nnz_total_dev_host_ptr,
                                                          void *temp_buffer);



alphasparse_status_t alphasparse_dcu_sprune_dense2csr(alphasparse_dcu_handle_t handle,
                                                      ALPHA_INT m,
                                                      ALPHA_INT n,
                                                      const float *A,
                                                      ALPHA_INT lda,
                                                      const float *threshold,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      float *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      ALPHA_INT *csr_col_ind,
                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_dense2csr(alphasparse_dcu_handle_t handle,
                                                      ALPHA_INT m,
                                                      ALPHA_INT n,
                                                      const double *A,
                                                      ALPHA_INT lda,
                                                      const double *threshold,
                                                      const alpha_dcu_matrix_descr_t descr,
                                                      double *csr_val,
                                                      const ALPHA_INT *csr_row_ptr,
                                                      ALPHA_INT *csr_col_ind,
                                                      void *temp_buffer);



alphasparse_status_t
alphasparse_dcu_sprune_dense2csr_by_percentage_buffer_size(alphasparse_dcu_handle_t handle,
                                                           ALPHA_INT m,
                                                           ALPHA_INT n,
                                                           const float *A,
                                                           ALPHA_INT lda,
                                                           float percentage,
                                                           const alpha_dcu_matrix_descr_t descr,
                                                           const float *csr_val,
                                                           const ALPHA_INT *csr_row_ptr,
                                                           const ALPHA_INT *csr_col_ind,
                                                           alphasparse_dcu_mat_info_t info,
                                                           size_t *buffer_size);

alphasparse_status_t
alphasparse_dcu_dprune_dense2csr_by_percentage_buffer_size(alphasparse_dcu_handle_t handle,
                                                           ALPHA_INT m,
                                                           ALPHA_INT n,
                                                           const double *A,
                                                           ALPHA_INT lda,
                                                           double percentage,
                                                           const alpha_dcu_matrix_descr_t descr,
                                                           const double *csr_val,
                                                           const ALPHA_INT *csr_row_ptr,
                                                           const ALPHA_INT *csr_col_ind,
                                                           alphasparse_dcu_mat_info_t info,
                                                           size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_sprune_dense2csr_nnz_by_percentage(alphasparse_dcu_handle_t handle,
                                                                        ALPHA_INT m,
                                                                        ALPHA_INT n,
                                                                        const float *A,
                                                                        ALPHA_INT lda,
                                                                        float percentage,
                                                                        const alpha_dcu_matrix_descr_t descr,
                                                                        ALPHA_INT *csr_row_ptr,
                                                                        ALPHA_INT *nnz_total_dev_host_ptr,
                                                                        alphasparse_dcu_mat_info_t info,
                                                                        void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_dense2csr_nnz_by_percentage(alphasparse_dcu_handle_t handle,
                                                                        ALPHA_INT m,
                                                                        ALPHA_INT n,
                                                                        const double *A,
                                                                        ALPHA_INT lda,
                                                                        double percentage,
                                                                        const alpha_dcu_matrix_descr_t descr,
                                                                        ALPHA_INT *csr_row_ptr,
                                                                        ALPHA_INT *nnz_total_dev_host_ptr,
                                                                        alphasparse_dcu_mat_info_t info,
                                                                        void *temp_buffer);



alphasparse_status_t alphasparse_dcu_sprune_dense2csr_by_percentage(alphasparse_dcu_handle_t handle,
                                                                    ALPHA_INT m,
                                                                    ALPHA_INT n,
                                                                    const float *A,
                                                                    ALPHA_INT lda,
                                                                    float percentage,
                                                                    const alpha_dcu_matrix_descr_t descr,
                                                                    float *csr_val,
                                                                    const ALPHA_INT *csr_row_ptr,
                                                                    ALPHA_INT *csr_col_ind,
                                                                    alphasparse_dcu_mat_info_t info,
                                                                    void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_dense2csr_by_percentage(alphasparse_dcu_handle_t handle,
                                                                    ALPHA_INT m,
                                                                    ALPHA_INT n,
                                                                    const double *A,
                                                                    ALPHA_INT lda,
                                                                    double percentage,
                                                                    const alpha_dcu_matrix_descr_t descr,
                                                                    double *csr_val,
                                                                    const ALPHA_INT *csr_row_ptr,
                                                                    ALPHA_INT *csr_col_ind,
                                                                    alphasparse_dcu_mat_info_t info,
                                                                    void *temp_buffer);



alphasparse_status_t alphasparse_dcu_sdense2csc(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const float *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_columns,
                                                float *csc_val,
                                                ALPHA_INT *csc_col_ptr,
                                                ALPHA_INT *csc_row_ind);

alphasparse_status_t alphasparse_dcu_ddense2csc(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const double *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_columns,
                                                double *csc_val,
                                                ALPHA_INT *csc_col_ptr,
                                                ALPHA_INT *csc_row_ind);

alphasparse_status_t alphasparse_dcu_cdense2csc(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex8 *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_columns,
                                                ALPHA_Complex8 *csc_val,
                                                ALPHA_INT *csc_col_ptr,
                                                ALPHA_INT *csc_row_ind);

alphasparse_status_t alphasparse_dcu_zdense2csc(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex16 *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_columns,
                                                ALPHA_Complex16 *csc_val,
                                                ALPHA_INT *csc_col_ptr,
                                                ALPHA_INT *csc_row_ind);



alphasparse_status_t alphasparse_dcu_sdense2coo(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const float *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                float *coo_val,
                                                ALPHA_INT *coo_row_ind,
                                                ALPHA_INT *coo_col_ind);

alphasparse_status_t alphasparse_dcu_ddense2coo(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const double *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                double *coo_val,
                                                ALPHA_INT *coo_row_ind,
                                                ALPHA_INT *coo_col_ind);

alphasparse_status_t alphasparse_dcu_cdense2coo(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex8 *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                ALPHA_Complex8 *coo_val,
                                                ALPHA_INT *coo_row_ind,
                                                ALPHA_INT *coo_col_ind);

alphasparse_status_t alphasparse_dcu_zdense2coo(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex16 *A,
                                                ALPHA_INT ld,
                                                const ALPHA_INT *nnz_per_rows,
                                                ALPHA_Complex16 *coo_val,
                                                ALPHA_INT *coo_row_ind,
                                                ALPHA_INT *coo_col_ind);



alphasparse_status_t alphasparse_dcu_scsr2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const float *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                float *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_dcsr2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_ccsr2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex8 *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_zcsr2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex16 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                ALPHA_Complex16 *A,
                                                ALPHA_INT ld);



alphasparse_status_t alphasparse_dcu_scsc2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const float *csc_val,
                                                const ALPHA_INT *csc_col_ptr,
                                                const ALPHA_INT *csc_row_ind,
                                                float *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_dcsc2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const double *csc_val,
                                                const ALPHA_INT *csc_col_ptr,
                                                const ALPHA_INT *csc_row_ind,
                                                double *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_ccsc2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex8 *csc_val,
                                                const ALPHA_INT *csc_col_ptr,
                                                const ALPHA_INT *csc_row_ind,
                                                ALPHA_Complex8 *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_zcsc2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex16 *csc_val,
                                                const ALPHA_INT *csc_col_ptr,
                                                const ALPHA_INT *csc_row_ind,
                                                ALPHA_Complex16 *A,
                                                ALPHA_INT ld);



alphasparse_status_t alphasparse_dcu_scoo2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT nnz,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const float *coo_val,
                                                const ALPHA_INT *coo_row_ind,
                                                const ALPHA_INT *coo_col_ind,
                                                float *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_dcoo2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT nnz,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const double *coo_val,
                                                const ALPHA_INT *coo_row_ind,
                                                const ALPHA_INT *coo_col_ind,
                                                double *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_ccoo2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT nnz,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex8 *coo_val,
                                                const ALPHA_INT *coo_row_ind,
                                                const ALPHA_INT *coo_col_ind,
                                                ALPHA_Complex8 *A,
                                                ALPHA_INT ld);

alphasparse_status_t alphasparse_dcu_zcoo2dense(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                ALPHA_INT nnz,
                                                const alpha_dcu_matrix_descr_t descr,
                                                const ALPHA_Complex16 *coo_val,
                                                const ALPHA_INT *coo_row_ind,
                                                const ALPHA_INT *coo_col_ind,
                                                ALPHA_Complex16 *A,
                                                ALPHA_INT ld);



alphasparse_status_t alphasparse_dcu_snnz_compress(alphasparse_dcu_handle_t handle,
                                                   ALPHA_INT m,
                                                   const alpha_dcu_matrix_descr_t descr_A,
                                                   const float *csr_val_A,
                                                   const ALPHA_INT *csr_row_ptr_A,
                                                   ALPHA_INT *nnz_per_row,
                                                   ALPHA_INT *nnz_C,
                                                   float tol);

alphasparse_status_t alphasparse_dcu_dnnz_compress(alphasparse_dcu_handle_t handle,
                                                   ALPHA_INT m,
                                                   const alpha_dcu_matrix_descr_t descr_A,
                                                   const double *csr_val_A,
                                                   const ALPHA_INT *csr_row_ptr_A,
                                                   ALPHA_INT *nnz_per_row,
                                                   ALPHA_INT *nnz_C,
                                                   double tol);

alphasparse_status_t alphasparse_dcu_cnnz_compress(alphasparse_dcu_handle_t handle,
                                                   ALPHA_INT m,
                                                   const alpha_dcu_matrix_descr_t descr_A,
                                                   const ALPHA_Complex8 *csr_val_A,
                                                   const ALPHA_INT *csr_row_ptr_A,
                                                   ALPHA_INT *nnz_per_row,
                                                   ALPHA_INT *nnz_C,
                                                   ALPHA_Complex8 tol);

alphasparse_status_t alphasparse_dcu_znnz_compress(alphasparse_dcu_handle_t handle,
                                                   ALPHA_INT m,
                                                   const alpha_dcu_matrix_descr_t descr_A,
                                                   const ALPHA_Complex16 *csr_val_A,
                                                   const ALPHA_INT *csr_row_ptr_A,
                                                   ALPHA_INT *nnz_per_row,
                                                   ALPHA_INT *nnz_C,
                                                   ALPHA_Complex16 tol);



alphasparse_status_t alphasparse_dcu_csr2coo(alphasparse_dcu_handle_t handle,
                                             const ALPHA_INT *csr_row_ptr,
                                             ALPHA_INT nnz,
                                             ALPHA_INT m,
                                             ALPHA_INT *coo_row_ind,
                                             alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_csr2csc_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT n,
                                                         ALPHA_INT nnz,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         alphasparse_dcu_action_t copy_values,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_scsr2csc(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT nnz,
                                              const float *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              float *csc_val,
                                              ALPHA_INT *csc_row_ind,
                                              ALPHA_INT *csc_col_ptr,
                                              alphasparse_dcu_action_t copy_values,
                                              alphasparse_index_base_t idx_base,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dcsr2csc(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT nnz,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              double *csc_val,
                                              ALPHA_INT *csc_row_ind,
                                              ALPHA_INT *csc_col_ptr,
                                              alphasparse_dcu_action_t copy_values,
                                              alphasparse_index_base_t idx_base,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_ccsr2csc(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT nnz,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              ALPHA_Complex8 *csc_val,
                                              ALPHA_INT *csc_row_ind,
                                              ALPHA_INT *csc_col_ptr,
                                              alphasparse_dcu_action_t copy_values,
                                              alphasparse_index_base_t idx_base,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zcsr2csc(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              ALPHA_INT nnz,
                                              const ALPHA_Complex16 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              ALPHA_Complex16 *csc_val,
                                              ALPHA_INT *csc_row_ind,
                                              ALPHA_INT *csc_col_ptr,
                                              alphasparse_dcu_action_t copy_values,
                                              alphasparse_index_base_t idx_base,
                                              void *temp_buffer);



alphasparse_status_t alphasparse_dcu_sgebsr2gebsc_buffer_size(alphasparse_dcu_handle_t handle,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const float *bsr_val,
                                                              const ALPHA_INT *bsr_row_ptr,
                                                              const ALPHA_INT *bsr_col_ind,
                                                              ALPHA_INT row_block_dim,
                                                              ALPHA_INT col_block_dim,
                                                              size_t *p_buffer_size);

alphasparse_status_t alphasparse_dcu_dgebsr2gebsc_buffer_size(alphasparse_dcu_handle_t handle,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const double *bsr_val,
                                                              const ALPHA_INT *bsr_row_ptr,
                                                              const ALPHA_INT *bsr_col_ind,
                                                              ALPHA_INT row_block_dim,
                                                              ALPHA_INT col_block_dim,
                                                              size_t *p_buffer_size);

alphasparse_status_t alphasparse_dcu_cgebsr2gebsc_buffer_size(alphasparse_dcu_handle_t handle,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const ALPHA_Complex8 *bsr_val,
                                                              const ALPHA_INT *bsr_row_ptr,
                                                              const ALPHA_INT *bsr_col_ind,
                                                              ALPHA_INT row_block_dim,
                                                              ALPHA_INT col_block_dim,
                                                              size_t *p_buffer_size);

alphasparse_status_t alphasparse_dcu_zgebsr2gebsc_buffer_size(alphasparse_dcu_handle_t handle,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const ALPHA_Complex16 *bsr_val,
                                                              const ALPHA_INT *bsr_row_ptr,
                                                              const ALPHA_INT *bsr_col_ind,
                                                              ALPHA_INT row_block_dim,
                                                              ALPHA_INT col_block_dim,
                                                              size_t *p_buffer_size);



alphasparse_status_t alphasparse_dcu_sgebsr2gebsc(alphasparse_dcu_handle_t handle,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const float *bsr_val,
                                                  const ALPHA_INT *bsr_row_ptr,
                                                  const ALPHA_INT *bsr_col_ind,
                                                  ALPHA_INT row_block_dim,
                                                  ALPHA_INT col_block_dim,
                                                  float *bsc_val,
                                                  ALPHA_INT *bsc_row_ind,
                                                  ALPHA_INT *bsc_col_ptr,
                                                  alphasparse_dcu_action_t copy_values,
                                                  alphasparse_index_base_t idx_base,
                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dgebsr2gebsc(alphasparse_dcu_handle_t handle,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const double *bsr_val,
                                                  const ALPHA_INT *bsr_row_ptr,
                                                  const ALPHA_INT *bsr_col_ind,
                                                  ALPHA_INT row_block_dim,
                                                  ALPHA_INT col_block_dim,
                                                  double *bsc_val,
                                                  ALPHA_INT *bsc_row_ind,
                                                  ALPHA_INT *bsc_col_ptr,
                                                  alphasparse_dcu_action_t copy_values,
                                                  alphasparse_index_base_t idx_base,
                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_cgebsr2gebsc(alphasparse_dcu_handle_t handle,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const ALPHA_Complex8 *bsr_val,
                                                  const ALPHA_INT *bsr_row_ptr,
                                                  const ALPHA_INT *bsr_col_ind,
                                                  ALPHA_INT row_block_dim,
                                                  ALPHA_INT col_block_dim,
                                                  ALPHA_Complex8 *bsc_val,
                                                  ALPHA_INT *bsc_row_ind,
                                                  ALPHA_INT *bsc_col_ptr,
                                                  alphasparse_dcu_action_t copy_values,
                                                  alphasparse_index_base_t idx_base,
                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zgebsr2gebsc(alphasparse_dcu_handle_t handle,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const ALPHA_Complex16 *bsr_val,
                                                  const ALPHA_INT *bsr_row_ptr,
                                                  const ALPHA_INT *bsr_col_ind,
                                                  ALPHA_INT row_block_dim,
                                                  ALPHA_INT col_block_dim,
                                                  ALPHA_Complex16 *bsc_val,
                                                  ALPHA_INT *bsc_row_ind,
                                                  ALPHA_INT *bsc_col_ptr,
                                                  alphasparse_dcu_action_t copy_values,
                                                  alphasparse_index_base_t idx_base,
                                                  void *temp_buffer);



alphasparse_status_t alphasparse_dcu_csr2ell_width(alphasparse_dcu_handle_t handle,
                                                   ALPHA_INT m,
                                                   const alpha_dcu_matrix_descr_t csr_descr,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const alpha_dcu_matrix_descr_t ell_descr,
                                                   ALPHA_INT *ell_width);



alphasparse_status_t alphasparse_dcu_scsr2ell(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const float *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              float *ell_val,
                                              ALPHA_INT *ell_col_ind);

alphasparse_status_t alphasparse_dcu_dcsr2ell(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              double *ell_val,
                                              ALPHA_INT *ell_col_ind);

alphasparse_status_t alphasparse_dcu_ccsr2ell(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              ALPHA_Complex8 *ell_val,
                                              ALPHA_INT *ell_col_ind);

alphasparse_status_t alphasparse_dcu_zcsr2ell(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const ALPHA_Complex16 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              ALPHA_Complex16 *ell_val,
                                              ALPHA_INT *ell_col_ind);



alphasparse_status_t alphasparse_dcu_scsr2hyb(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const float *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_hyb_mat_t hyb,
                                              ALPHA_INT user_ell_width,
                                              alphasparse_dcu_hyb_partition_t partition_type);

alphasparse_status_t alphasparse_dcu_dcsr2hyb(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_hyb_mat_t hyb,
                                              ALPHA_INT user_ell_width,
                                              alphasparse_dcu_hyb_partition_t partition_type);

alphasparse_status_t alphasparse_dcu_ccsr2hyb(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_hyb_mat_t hyb,
                                              ALPHA_INT user_ell_width,
                                              alphasparse_dcu_hyb_partition_t partition_type);

alphasparse_status_t alphasparse_dcu_zcsr2hyb(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const ALPHA_Complex16 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_hyb_mat_t hyb,
                                              ALPHA_INT user_ell_width,
                                              alphasparse_dcu_hyb_partition_t partition_type);



alphasparse_status_t alphasparse_dcu_csr2bsr_nnz(alphasparse_dcu_handle_t handle,
                                                 alphasparse_layout_t dir,
                                                 ALPHA_INT m,
                                                 ALPHA_INT n,
                                                 const alpha_dcu_matrix_descr_t csr_descr,
                                                 const ALPHA_INT *csr_row_ptr,
                                                 const ALPHA_INT *csr_col_ind,
                                                 ALPHA_INT block_dim,
                                                 const alpha_dcu_matrix_descr_t bsr_descr,
                                                 ALPHA_INT *bsr_row_ptr,
                                                 ALPHA_INT *bsr_nnz);



alphasparse_status_t alphasparse_dcu_scsr2bsr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const float *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              float *bsr_val,
                                              ALPHA_INT *bsr_row_ptr,
                                              ALPHA_INT *bsr_col_ind);

alphasparse_status_t alphasparse_dcu_dcsr2bsr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              double *bsr_val,
                                              ALPHA_INT *bsr_row_ptr,
                                              ALPHA_INT *bsr_col_ind);

alphasparse_status_t alphasparse_dcu_ccsr2bsr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              ALPHA_Complex8 *bsr_val,
                                              ALPHA_INT *bsr_row_ptr,
                                              ALPHA_INT *bsr_col_ind);

alphasparse_status_t alphasparse_dcu_zcsr2bsr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              const ALPHA_Complex16 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              ALPHA_Complex16 *bsr_val,
                                              ALPHA_INT *bsr_row_ptr,
                                              ALPHA_INT *bsr_col_ind);



alphasparse_status_t alphasparse_dcu_scsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                            alphasparse_layout_t dir,
                                                            ALPHA_INT m,
                                                            ALPHA_INT n,
                                                            const alpha_dcu_matrix_descr_t csr_descr,
                                                            const float *csr_val,
                                                            const ALPHA_INT *csr_row_ptr,
                                                            const ALPHA_INT *csr_col_ind,
                                                            ALPHA_INT row_block_dim,
                                                            ALPHA_INT col_block_dim,
                                                            size_t *p_buffer_size);

alphasparse_status_t alphasparse_dcu_dcsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                            alphasparse_layout_t dir,
                                                            ALPHA_INT m,
                                                            ALPHA_INT n,
                                                            const alpha_dcu_matrix_descr_t csr_descr,
                                                            const double *csr_val,
                                                            const ALPHA_INT *csr_row_ptr,
                                                            const ALPHA_INT *csr_col_ind,
                                                            ALPHA_INT row_block_dim,
                                                            ALPHA_INT col_block_dim,
                                                            size_t *p_buffer_size);

alphasparse_status_t alphasparse_dcu_ccsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                            alphasparse_layout_t dir,
                                                            ALPHA_INT m,
                                                            ALPHA_INT n,
                                                            const alpha_dcu_matrix_descr_t csr_descr,
                                                            const ALPHA_Complex8 *csr_val,
                                                            const ALPHA_INT *csr_row_ptr,
                                                            const ALPHA_INT *csr_col_ind,
                                                            ALPHA_INT row_block_dim,
                                                            ALPHA_INT col_block_dim,
                                                            size_t *p_buffer_size);

alphasparse_status_t alphasparse_dcu_zcsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                            alphasparse_layout_t dir,
                                                            ALPHA_INT m,
                                                            ALPHA_INT n,
                                                            const alpha_dcu_matrix_descr_t csr_descr,
                                                            const ALPHA_Complex16 *csr_val,
                                                            const ALPHA_INT *csr_row_ptr,
                                                            const ALPHA_INT *csr_col_ind,
                                                            ALPHA_INT row_block_dim,
                                                            ALPHA_INT col_block_dim,
                                                            size_t *p_buffer_size);



alphasparse_status_t alphasparse_dcu_csr2gebsr_nnz(alphasparse_dcu_handle_t handle,
                                                   alphasparse_layout_t dir,
                                                   ALPHA_INT m,
                                                   ALPHA_INT n,
                                                   const alpha_dcu_matrix_descr_t csr_descr,
                                                   const ALPHA_INT *csr_row_ptr,
                                                   const ALPHA_INT *csr_col_ind,
                                                   const alpha_dcu_matrix_descr_t bsr_descr,
                                                   ALPHA_INT *bsr_row_ptr,
                                                   ALPHA_INT row_block_dim,
                                                   ALPHA_INT col_block_dim,
                                                   ALPHA_INT *bsr_nnz_devhost,
                                                   void *p_buffer);



alphasparse_status_t alphasparse_dcu_scsr2gebsr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                const float *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                float *bsr_val,
                                                ALPHA_INT *bsr_row_ptr,
                                                ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                void *p_buffer);

alphasparse_status_t alphasparse_dcu_dcsr2gebsr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                double *bsr_val,
                                                ALPHA_INT *bsr_row_ptr,
                                                ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                void *p_buffer);

alphasparse_status_t alphasparse_dcu_ccsr2gebsr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                const ALPHA_Complex8 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                ALPHA_Complex8 *bsr_val,
                                                ALPHA_INT *bsr_row_ptr,
                                                ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                void *p_buffer);

alphasparse_status_t alphasparse_dcu_zcsr2gebsr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT m,
                                                ALPHA_INT n,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                const ALPHA_Complex16 *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                ALPHA_Complex16 *bsr_val,
                                                ALPHA_INT *bsr_row_ptr,
                                                ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                void *p_buffer);



alphasparse_status_t alphasparse_dcu_scsr2csr_compress(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT n,
                                                       const alpha_dcu_matrix_descr_t descr_A,
                                                       const float *csr_val_A,
                                                       const ALPHA_INT *csr_row_ptr_A,
                                                       const ALPHA_INT *csr_col_ind_A,
                                                       ALPHA_INT nnz_A,
                                                       const ALPHA_INT *nnz_per_row,
                                                       float *csr_val_C,
                                                       ALPHA_INT *csr_row_ptr_C,
                                                       ALPHA_INT *csr_col_ind_C,
                                                       float tol);

alphasparse_status_t alphasparse_dcu_dcsr2csr_compress(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT n,
                                                       const alpha_dcu_matrix_descr_t descr_A,
                                                       const double *csr_val_A,
                                                       const ALPHA_INT *csr_row_ptr_A,
                                                       const ALPHA_INT *csr_col_ind_A,
                                                       ALPHA_INT nnz_A,
                                                       const ALPHA_INT *nnz_per_row,
                                                       double *csr_val_C,
                                                       ALPHA_INT *csr_row_ptr_C,
                                                       ALPHA_INT *csr_col_ind_C,
                                                       double tol);

alphasparse_status_t alphasparse_dcu_ccsr2csr_compress(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT n,
                                                       const alpha_dcu_matrix_descr_t descr_A,
                                                       const ALPHA_Complex8 *csr_val_A,
                                                       const ALPHA_INT *csr_row_ptr_A,
                                                       const ALPHA_INT *csr_col_ind_A,
                                                       ALPHA_INT nnz_A,
                                                       const ALPHA_INT *nnz_per_row,
                                                       ALPHA_Complex8 *csr_val_C,
                                                       ALPHA_INT *csr_row_ptr_C,
                                                       ALPHA_INT *csr_col_ind_C,
                                                       ALPHA_Complex8 tol);

alphasparse_status_t alphasparse_dcu_zcsr2csr_compress(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT n,
                                                       const alpha_dcu_matrix_descr_t descr_A,
                                                       const ALPHA_Complex16 *csr_val_A,
                                                       const ALPHA_INT *csr_row_ptr_A,
                                                       const ALPHA_INT *csr_col_ind_A,
                                                       ALPHA_INT nnz_A,
                                                       const ALPHA_INT *nnz_per_row,
                                                       ALPHA_Complex16 *csr_val_C,
                                                       ALPHA_INT *csr_row_ptr_C,
                                                       ALPHA_INT *csr_col_ind_C,
                                                       ALPHA_Complex16 tol);



alphasparse_status_t alphasparse_dcu_sprune_csr2csr_buffer_size(alphasparse_dcu_handle_t handle,
                                                                ALPHA_INT m,
                                                                ALPHA_INT n,
                                                                ALPHA_INT nnz_A,
                                                                const alpha_dcu_matrix_descr_t csr_descr_A,
                                                                const float *csr_val_A,
                                                                const ALPHA_INT *csr_row_ptr_A,
                                                                const ALPHA_INT *csr_col_ind_A,
                                                                const float *threshold,
                                                                const alpha_dcu_matrix_descr_t csr_descr_C,
                                                                const float *csr_val_C,
                                                                const ALPHA_INT *csr_row_ptr_C,
                                                                const ALPHA_INT *csr_col_ind_C,
                                                                size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dprune_csr2csr_buffer_size(alphasparse_dcu_handle_t handle,
                                                                ALPHA_INT m,
                                                                ALPHA_INT n,
                                                                ALPHA_INT nnz_A,
                                                                const alpha_dcu_matrix_descr_t csr_descr_A,
                                                                const double *csr_val_A,
                                                                const ALPHA_INT *csr_row_ptr_A,
                                                                const ALPHA_INT *csr_col_ind_A,
                                                                const double *threshold,
                                                                const alpha_dcu_matrix_descr_t csr_descr_C,
                                                                const double *csr_val_C,
                                                                const ALPHA_INT *csr_row_ptr_C,
                                                                const ALPHA_INT *csr_col_ind_C,
                                                                size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_sprune_csr2csr_nnz(alphasparse_dcu_handle_t handle,
                                                        ALPHA_INT m,
                                                        ALPHA_INT n,
                                                        ALPHA_INT nnz_A,
                                                        const alpha_dcu_matrix_descr_t csr_descr_A,
                                                        const float *csr_val_A,
                                                        const ALPHA_INT *csr_row_ptr_A,
                                                        const ALPHA_INT *csr_col_ind_A,
                                                        const float *threshold,
                                                        const alpha_dcu_matrix_descr_t csr_descr_C,
                                                        ALPHA_INT *csr_row_ptr_C,
                                                        ALPHA_INT *nnz_total_dev_host_ptr,
                                                        void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_csr2csr_nnz(alphasparse_dcu_handle_t handle,
                                                        ALPHA_INT m,
                                                        ALPHA_INT n,
                                                        ALPHA_INT nnz_A,
                                                        const alpha_dcu_matrix_descr_t csr_descr_A,
                                                        const double *csr_val_A,
                                                        const ALPHA_INT *csr_row_ptr_A,
                                                        const ALPHA_INT *csr_col_ind_A,
                                                        const double *threshold,
                                                        const alpha_dcu_matrix_descr_t csr_descr_C,
                                                        ALPHA_INT *csr_row_ptr_C,
                                                        ALPHA_INT *nnz_total_dev_host_ptr,
                                                        void *temp_buffer);



alphasparse_status_t alphasparse_dcu_sprune_csr2csr(alphasparse_dcu_handle_t handle,
                                                    ALPHA_INT m,
                                                    ALPHA_INT n,
                                                    ALPHA_INT nnz_A,
                                                    const alpha_dcu_matrix_descr_t csr_descr_A,
                                                    const float *csr_val_A,
                                                    const ALPHA_INT *csr_row_ptr_A,
                                                    const ALPHA_INT *csr_col_ind_A,
                                                    const float *threshold,
                                                    const alpha_dcu_matrix_descr_t csr_descr_C,
                                                    float *csr_val_C,
                                                    const ALPHA_INT *csr_row_ptr_C,
                                                    ALPHA_INT *csr_col_ind_C,
                                                    void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_csr2csr(alphasparse_dcu_handle_t handle,
                                                    ALPHA_INT m,
                                                    ALPHA_INT n,
                                                    ALPHA_INT nnz_A,
                                                    const alpha_dcu_matrix_descr_t csr_descr_A,
                                                    const double *csr_val_A,
                                                    const ALPHA_INT *csr_row_ptr_A,
                                                    const ALPHA_INT *csr_col_ind_A,
                                                    const double *threshold,
                                                    const alpha_dcu_matrix_descr_t csr_descr_C,
                                                    double *csr_val_C,
                                                    const ALPHA_INT *csr_row_ptr_C,
                                                    ALPHA_INT *csr_col_ind_C,
                                                    void *temp_buffer);



alphasparse_status_t
alphasparse_dcu_sprune_csr2csr_by_percentage_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT n,
                                                         ALPHA_INT nnz_A,
                                                         const alpha_dcu_matrix_descr_t csr_descr_A,
                                                         const float *csr_val_A,
                                                         const ALPHA_INT *csr_row_ptr_A,
                                                         const ALPHA_INT *csr_col_ind_A,
                                                         float percentage,
                                                         const alpha_dcu_matrix_descr_t csr_descr_C,
                                                         const float *csr_val_C,
                                                         const ALPHA_INT *csr_row_ptr_C,
                                                         const ALPHA_INT *csr_col_ind_C,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);

alphasparse_status_t
alphasparse_dcu_dprune_csr2csr_by_percentage_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT n,
                                                         ALPHA_INT nnz_A,
                                                         const alpha_dcu_matrix_descr_t csr_descr_A,
                                                         const double *csr_val_A,
                                                         const ALPHA_INT *csr_row_ptr_A,
                                                         const ALPHA_INT *csr_col_ind_A,
                                                         double percentage,
                                                         const alpha_dcu_matrix_descr_t csr_descr_C,
                                                         const double *csr_val_C,
                                                         const ALPHA_INT *csr_row_ptr_C,
                                                         const ALPHA_INT *csr_col_ind_C,
                                                         alphasparse_dcu_mat_info_t info,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_sprune_csr2csr_nnz_by_percentage(alphasparse_dcu_handle_t handle,
                                                                      ALPHA_INT m,
                                                                      ALPHA_INT n,
                                                                      ALPHA_INT nnz_A,
                                                                      const alpha_dcu_matrix_descr_t csr_descr_A,
                                                                      const float *csr_val_A,
                                                                      const ALPHA_INT *csr_row_ptr_A,
                                                                      const ALPHA_INT *csr_col_ind_A,
                                                                      float percentage,
                                                                      const alpha_dcu_matrix_descr_t csr_descr_C,
                                                                      ALPHA_INT *csr_row_ptr_C,
                                                                      ALPHA_INT *nnz_total_dev_host_ptr,
                                                                      alphasparse_dcu_mat_info_t info,
                                                                      void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_csr2csr_nnz_by_percentage(alphasparse_dcu_handle_t handle,
                                                                      ALPHA_INT m,
                                                                      ALPHA_INT n,
                                                                      ALPHA_INT nnz_A,
                                                                      const alpha_dcu_matrix_descr_t csr_descr_A,
                                                                      const double *csr_val_A,
                                                                      const ALPHA_INT *csr_row_ptr_A,
                                                                      const ALPHA_INT *csr_col_ind_A,
                                                                      double percentage,
                                                                      const alpha_dcu_matrix_descr_t csr_descr_C,
                                                                      ALPHA_INT *csr_row_ptr_C,
                                                                      ALPHA_INT *nnz_total_dev_host_ptr,
                                                                      alphasparse_dcu_mat_info_t info,
                                                                      void *temp_buffer);



alphasparse_status_t alphasparse_dcu_sprune_csr2csr_by_percentage(alphasparse_dcu_handle_t handle,
                                                                  ALPHA_INT m,
                                                                  ALPHA_INT n,
                                                                  ALPHA_INT nnz_A,
                                                                  const alpha_dcu_matrix_descr_t csr_descr_A,
                                                                  const float *csr_val_A,
                                                                  const ALPHA_INT *csr_row_ptr_A,
                                                                  const ALPHA_INT *csr_col_ind_A,
                                                                  float percentage,
                                                                  const alpha_dcu_matrix_descr_t csr_descr_C,
                                                                  float *csr_val_C,
                                                                  const ALPHA_INT *csr_row_ptr_C,
                                                                  ALPHA_INT *csr_col_ind_C,
                                                                  alphasparse_dcu_mat_info_t info,
                                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dprune_csr2csr_by_percentage(alphasparse_dcu_handle_t handle,
                                                                  ALPHA_INT m,
                                                                  ALPHA_INT n,
                                                                  ALPHA_INT nnz_A,
                                                                  const alpha_dcu_matrix_descr_t csr_descr_A,
                                                                  const double *csr_val_A,
                                                                  const ALPHA_INT *csr_row_ptr_A,
                                                                  const ALPHA_INT *csr_col_ind_A,
                                                                  double percentage,
                                                                  const alpha_dcu_matrix_descr_t csr_descr_C,
                                                                  double *csr_val_C,
                                                                  const ALPHA_INT *csr_row_ptr_C,
                                                                  ALPHA_INT *csr_col_ind_C,
                                                                  alphasparse_dcu_mat_info_t info,
                                                                  void *temp_buffer);



alphasparse_status_t alphasparse_dcu_coo2csr(alphasparse_dcu_handle_t handle,
                                             const ALPHA_INT *coo_row_ind,
                                             ALPHA_INT nnz,
                                             ALPHA_INT m,
                                             ALPHA_INT *csr_row_ptr,
                                             alphasparse_index_base_t idx_base);



alphasparse_status_t alphasparse_dcu_ell2csr_nnz(alphasparse_dcu_handle_t handle,
                                                 ALPHA_INT m,
                                                 ALPHA_INT n,
                                                 const alpha_dcu_matrix_descr_t ell_descr,
                                                 ALPHA_INT ell_width,
                                                 const ALPHA_INT *ell_col_ind,
                                                 const alpha_dcu_matrix_descr_t csr_descr,
                                                 ALPHA_INT *csr_row_ptr,
                                                 ALPHA_INT *csr_nnz);



alphasparse_status_t alphasparse_dcu_sell2csr(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              const float *ell_val,
                                              const ALPHA_INT *ell_col_ind,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              float *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_dell2csr(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              const double *ell_val,
                                              const ALPHA_INT *ell_col_ind,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_cell2csr(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              const ALPHA_Complex8 *ell_val,
                                              const ALPHA_INT *ell_col_ind,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              ALPHA_Complex8 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_zell2csr(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT n,
                                              const alpha_dcu_matrix_descr_t ell_descr,
                                              ALPHA_INT ell_width,
                                              const ALPHA_Complex16 *ell_val,
                                              const ALPHA_INT *ell_col_ind,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              ALPHA_Complex16 *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);



alphasparse_status_t alphasparse_dcu_hyb2csr_buffer_size(alphasparse_dcu_handle_t handle,
                                                         const alpha_dcu_matrix_descr_t descr,
                                                         const alphasparse_dcu_hyb_mat_t hyb,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_shyb2csr(alphasparse_dcu_handle_t handle,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const alphasparse_dcu_hyb_mat_t hyb,
                                              float *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dhyb2csr(alphasparse_dcu_handle_t handle,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const alphasparse_dcu_hyb_mat_t hyb,
                                              double *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_chyb2csr(alphasparse_dcu_handle_t handle,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const alphasparse_dcu_hyb_mat_t hyb,
                                              ALPHA_Complex8 *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind,
                                              void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zhyb2csr(alphasparse_dcu_handle_t handle,
                                              const alpha_dcu_matrix_descr_t descr,
                                              const alphasparse_dcu_hyb_mat_t hyb,
                                              ALPHA_Complex16 *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind,
                                              void *temp_buffer);



alphasparse_status_t alphasparse_dcu_create_identity_permutation(alphasparse_dcu_handle_t handle,
                                                                 ALPHA_INT n,
                                                                 ALPHA_INT *p);



alphasparse_status_t alphasparse_dcu_csrsort_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT n,
                                                         ALPHA_INT nnz,
                                                         const ALPHA_INT *csr_row_ptr,
                                                         const ALPHA_INT *csr_col_ind,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_csrsort(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_INT *csr_row_ptr,
                                             ALPHA_INT *csr_col_ind,
                                             ALPHA_INT *perm,
                                             void *temp_buffer);



alphasparse_status_t alphasparse_dcu_cscsort_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT n,
                                                         ALPHA_INT nnz,
                                                         const ALPHA_INT *csc_col_ptr,
                                                         const ALPHA_INT *csc_row_ind,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_cscsort(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT nnz,
                                             const alpha_dcu_matrix_descr_t descr,
                                             const ALPHA_INT *csc_col_ptr,
                                             ALPHA_INT *csc_row_ind,
                                             ALPHA_INT *perm,
                                             void *temp_buffer);



alphasparse_status_t alphasparse_dcu_coosort_buffer_size(alphasparse_dcu_handle_t handle,
                                                         ALPHA_INT m,
                                                         ALPHA_INT n,
                                                         ALPHA_INT nnz,
                                                         const ALPHA_INT *coo_row_ind,
                                                         const ALPHA_INT *coo_col_ind,
                                                         size_t *buffer_size);



alphasparse_status_t alphasparse_dcu_coosort_by_row(alphasparse_dcu_handle_t handle,
                                                    ALPHA_INT m,
                                                    ALPHA_INT n,
                                                    ALPHA_INT nnz,
                                                    ALPHA_INT *coo_row_ind,
                                                    ALPHA_INT *coo_col_ind,
                                                    ALPHA_INT *perm,
                                                    void *temp_buffer);



alphasparse_status_t alphasparse_dcu_coosort_by_column(alphasparse_dcu_handle_t handle,
                                                       ALPHA_INT m,
                                                       ALPHA_INT n,
                                                       ALPHA_INT nnz,
                                                       ALPHA_INT *coo_row_ind,
                                                       ALPHA_INT *coo_col_ind,
                                                       ALPHA_INT *perm,
                                                       void *temp_buffer);



alphasparse_status_t alphasparse_dcu_sbsr2csr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nb,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              const float *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              float *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_dbsr2csr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nb,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              const double *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              double *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_cbsr2csr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nb,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              const ALPHA_Complex8 *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              ALPHA_Complex8 *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_zbsr2csr(alphasparse_dcu_handle_t handle,
                                              alphasparse_layout_t dir,
                                              ALPHA_INT mb,
                                              ALPHA_INT nb,
                                              const alpha_dcu_matrix_descr_t bsr_descr,
                                              const ALPHA_Complex16 *bsr_val,
                                              const ALPHA_INT *bsr_row_ptr,
                                              const ALPHA_INT *bsr_col_ind,
                                              ALPHA_INT block_dim,
                                              const alpha_dcu_matrix_descr_t csr_descr,
                                              ALPHA_Complex16 *csr_val,
                                              ALPHA_INT *csr_row_ptr,
                                              ALPHA_INT *csr_col_ind);



alphasparse_status_t alphasparse_dcu_sgebsr2csr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT mb,
                                                ALPHA_INT nb,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                const float *bsr_val,
                                                const ALPHA_INT *bsr_row_ptr,
                                                const ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                float *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_dgebsr2csr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT mb,
                                                ALPHA_INT nb,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                const double *bsr_val,
                                                const ALPHA_INT *bsr_row_ptr,
                                                const ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                double *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_cgebsr2csr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT mb,
                                                ALPHA_INT nb,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                const ALPHA_Complex8 *bsr_val,
                                                const ALPHA_INT *bsr_row_ptr,
                                                const ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                ALPHA_Complex8 *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_zgebsr2csr(alphasparse_dcu_handle_t handle,
                                                alphasparse_layout_t dir,
                                                ALPHA_INT mb,
                                                ALPHA_INT nb,
                                                const alpha_dcu_matrix_descr_t bsr_descr,
                                                const ALPHA_Complex16 *bsr_val,
                                                const ALPHA_INT *bsr_row_ptr,
                                                const ALPHA_INT *bsr_col_ind,
                                                ALPHA_INT row_block_dim,
                                                ALPHA_INT col_block_dim,
                                                const alpha_dcu_matrix_descr_t csr_descr,
                                                ALPHA_Complex16 *csr_val,
                                                ALPHA_INT *csr_row_ptr,
                                                ALPHA_INT *csr_col_ind);

alphasparse_status_t alphasparse_dcu_sgebsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                              alphasparse_layout_t dir,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const alpha_dcu_matrix_descr_t descr_A,
                                                              const float *bsr_val_A,
                                                              const ALPHA_INT *bsr_row_ptr_A,
                                                              const ALPHA_INT *bsr_col_ind_A,
                                                              ALPHA_INT row_block_dim_A,
                                                              ALPHA_INT col_block_dim_A,
                                                              ALPHA_INT row_block_dim_C,
                                                              ALPHA_INT col_block_dim_C,
                                                              size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_dgebsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                              alphasparse_layout_t dir,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const alpha_dcu_matrix_descr_t descr_A,
                                                              const double *bsr_val_A,
                                                              const ALPHA_INT *bsr_row_ptr_A,
                                                              const ALPHA_INT *bsr_col_ind_A,
                                                              ALPHA_INT row_block_dim_A,
                                                              ALPHA_INT col_block_dim_A,
                                                              ALPHA_INT row_block_dim_C,
                                                              ALPHA_INT col_block_dim_C,
                                                              size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_cgebsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                              alphasparse_layout_t dir,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const alpha_dcu_matrix_descr_t descr_A,
                                                              const ALPHA_Complex8 *bsr_val_A,
                                                              const ALPHA_INT *bsr_row_ptr_A,
                                                              const ALPHA_INT *bsr_col_ind_A,
                                                              ALPHA_INT row_block_dim_A,
                                                              ALPHA_INT col_block_dim_A,
                                                              ALPHA_INT row_block_dim_C,
                                                              ALPHA_INT col_block_dim_C,
                                                              size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_zgebsr2gebsr_buffer_size(alphasparse_dcu_handle_t handle,
                                                              alphasparse_layout_t dir,
                                                              ALPHA_INT mb,
                                                              ALPHA_INT nb,
                                                              ALPHA_INT nnzb,
                                                              const alpha_dcu_matrix_descr_t descr_A,
                                                              const ALPHA_Complex16 *bsr_val_A,
                                                              const ALPHA_INT *bsr_row_ptr_A,
                                                              const ALPHA_INT *bsr_col_ind_A,
                                                              ALPHA_INT row_block_dim_A,
                                                              ALPHA_INT col_block_dim_A,
                                                              ALPHA_INT row_block_dim_C,
                                                              ALPHA_INT col_block_dim_C,
                                                              size_t *buffer_size);

alphasparse_status_t alphasparse_dcu_gebsr2gebsr_nnz(alphasparse_dcu_handle_t handle,
                                                     alphasparse_layout_t dir,
                                                     ALPHA_INT mb,
                                                     ALPHA_INT nb,
                                                     ALPHA_INT nnzb,
                                                     const alpha_dcu_matrix_descr_t descr_A,
                                                     const ALPHA_INT *bsr_row_ptr_A,
                                                     const ALPHA_INT *bsr_col_ind_A,
                                                     ALPHA_INT row_block_dim_A,
                                                     ALPHA_INT col_block_dim_A,
                                                     const alpha_dcu_matrix_descr_t descr_C,
                                                     ALPHA_INT *bsr_row_ptr_C,
                                                     ALPHA_INT row_block_dim_C,
                                                     ALPHA_INT col_block_dim_C,
                                                     ALPHA_INT *nnz_total_dev_host_ptr,
                                                     void *temp_buffer);

alphasparse_status_t alphasparse_dcu_sgebsr2gebsr(alphasparse_dcu_handle_t handle,
                                                  alphasparse_layout_t dir,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const alpha_dcu_matrix_descr_t descr_A,
                                                  const float *bsr_val_A,
                                                  const ALPHA_INT *bsr_row_ptr_A,
                                                  const ALPHA_INT *bsr_col_ind_A,
                                                  ALPHA_INT row_block_dim_A,
                                                  ALPHA_INT col_block_dim_A,
                                                  const alpha_dcu_matrix_descr_t descr_C,
                                                  float *bsr_val_C,
                                                  ALPHA_INT *bsr_row_ptr_C,
                                                  ALPHA_INT *bsr_col_ind_C,
                                                  ALPHA_INT row_block_dim_C,
                                                  ALPHA_INT col_block_dim_C,
                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dgebsr2gebsr(alphasparse_dcu_handle_t handle,
                                                  alphasparse_layout_t dir,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const alpha_dcu_matrix_descr_t descr_A,
                                                  const double *bsr_val_A,
                                                  const ALPHA_INT *bsr_row_ptr_A,
                                                  const ALPHA_INT *bsr_col_ind_A,
                                                  ALPHA_INT row_block_dim_A,
                                                  ALPHA_INT col_block_dim_A,
                                                  const alpha_dcu_matrix_descr_t descr_C,
                                                  double *bsr_val_C,
                                                  ALPHA_INT *bsr_row_ptr_C,
                                                  ALPHA_INT *bsr_col_ind_C,
                                                  ALPHA_INT row_block_dim_C,
                                                  ALPHA_INT col_block_dim_C,
                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_cgebsr2gebsr(alphasparse_dcu_handle_t handle,
                                                  alphasparse_layout_t dir,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const alpha_dcu_matrix_descr_t descr_A,
                                                  const ALPHA_Complex8 *bsr_val_A,
                                                  const ALPHA_INT *bsr_row_ptr_A,
                                                  const ALPHA_INT *bsr_col_ind_A,
                                                  ALPHA_INT row_block_dim_A,
                                                  ALPHA_INT col_block_dim_A,
                                                  const alpha_dcu_matrix_descr_t descr_C,
                                                  ALPHA_Complex8 *bsr_val_C,
                                                  ALPHA_INT *bsr_row_ptr_C,
                                                  ALPHA_INT *bsr_col_ind_C,
                                                  ALPHA_INT row_block_dim_C,
                                                  ALPHA_INT col_block_dim_C,
                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_zgebsr2gebsr(alphasparse_dcu_handle_t handle,
                                                  alphasparse_layout_t dir,
                                                  ALPHA_INT mb,
                                                  ALPHA_INT nb,
                                                  ALPHA_INT nnzb,
                                                  const alpha_dcu_matrix_descr_t descr_A,
                                                  const ALPHA_Complex16 *bsr_val_A,
                                                  const ALPHA_INT *bsr_row_ptr_A,
                                                  const ALPHA_INT *bsr_col_ind_A,
                                                  ALPHA_INT row_block_dim_A,
                                                  ALPHA_INT col_block_dim_A,
                                                  const alpha_dcu_matrix_descr_t descr_C,
                                                  ALPHA_Complex16 *bsr_val_C,
                                                  ALPHA_INT *bsr_row_ptr_C,
                                                  ALPHA_INT *bsr_col_ind_C,
                                                  ALPHA_INT row_block_dim_C,
                                                  ALPHA_INT col_block_dim_C,
                                                  void *temp_buffer);

alphasparse_status_t alphasparse_dcu_axpby(alphasparse_dcu_handle_t handle,
                                           const void *alpha,
                                           const alphasparse_dcu_spvec_descr_t x,
                                           const void *beta,
                                           alphasparse_dcu_dnvec_descr_t y);

alphasparse_status_t alphasparse_dcu_gather(alphasparse_dcu_handle_t handle,
                                            const alphasparse_dcu_dnvec_descr_t y,
                                            alphasparse_dcu_spvec_descr_t x);

alphasparse_status_t alphasparse_dcu_scatter(alphasparse_dcu_handle_t handle,
                                             const alphasparse_dcu_spvec_descr_t x,
                                             alphasparse_dcu_dnvec_descr_t y);

alphasparse_status_t alphasparse_dcu_rot(alphasparse_dcu_handle_t handle,
                                         const void *c,
                                         const void *s,
                                         alphasparse_dcu_spvec_descr_t x,
                                         alphasparse_dcu_dnvec_descr_t y);

alphasparse_status_t alphasparse_dcu_sparse_to_dense(alphasparse_dcu_handle_t handle,
                                                     const alphasparse_dcu_spmat_descr_t mat_A,
                                                     alphasparse_dcu_dnmat_descr_t mat_B,
                                                     alphasparse_dcu_sparse_to_dense_alg_t alg,
                                                     size_t *buffer_size,
                                                     void *temp_buffer);

alphasparse_status_t alphasparse_dcu_dense_to_sparse(alphasparse_dcu_handle_t handle,
                                                     const alphasparse_dcu_dnmat_descr_t mat_A,
                                                     alphasparse_dcu_spmat_descr_t mat_B,
                                                     alphasparse_dcu_dense_to_sparse_alg_t alg,
                                                     size_t *buffer_size,
                                                     void *temp_buffer);

alphasparse_status_t alphasparse_dcu_spvv(alphasparse_dcu_handle_t handle,
                                          alphasparse_operation_t trans,
                                          const alphasparse_dcu_spvec_descr_t x,
                                          const alphasparse_dcu_dnvec_descr_t y,
                                          void *result,
                                          alphasparse_datatype_t compute_type,
                                          size_t *buffer_size,
                                          void *temp_buffer);

alphasparse_status_t alphasparse_dcu_spmv(alphasparse_dcu_handle_t handle,
                                          alphasparse_operation_t trans,
                                          const void *alpha,
                                          const alphasparse_dcu_spmat_descr_t mat,
                                          const alphasparse_dcu_dnvec_descr_t x,
                                          const void *beta,
                                          const alphasparse_dcu_dnvec_descr_t y,
                                          alphasparse_datatype_t compute_type,
                                          alphasparse_dcu_spmv_alg_t alg,
                                          size_t *buffer_size,
                                          void *temp_buffer);

alphasparse_status_t alphasparse_dcu_spgemm(alphasparse_dcu_handle_t handle,
                                            alphasparse_operation_t trans_A,
                                            alphasparse_operation_t trans_B,
                                            const void *alpha,
                                            const alphasparse_dcu_spmat_descr_t A,
                                            const alphasparse_dcu_spmat_descr_t B,
                                            const void *beta,
                                            const alphasparse_dcu_spmat_descr_t D,
                                            alphasparse_dcu_spmat_descr_t C,
                                            alphasparse_datatype_t compute_type,
                                            alphasparse_dcu_spgemm_alg_t alg,
                                            alphasparse_dcu_spgemm_stage_t stage,
                                            size_t *buffer_size,
                                            void *temp_buffer);

/**
 * convert from host point to device point
 * 
 */
alphasparse_status_t host2device_s_csr(alphasparse_matrix_t A);
alphasparse_status_t host2device_d_csr(alphasparse_matrix_t A);
alphasparse_status_t host2device_c_csr(alphasparse_matrix_t A);
alphasparse_status_t host2device_z_csr(alphasparse_matrix_t A);

alphasparse_status_t host2device_s_csr5(alphasparse_matrix_t A);
alphasparse_status_t host2device_d_csr5(alphasparse_matrix_t A);
alphasparse_status_t host2device_c_csr5(alphasparse_matrix_t A);
alphasparse_status_t host2device_z_csr5(alphasparse_matrix_t A);

alphasparse_status_t host2device_s_coo(alphasparse_matrix_t A);
alphasparse_status_t host2device_d_coo(alphasparse_matrix_t A);
alphasparse_status_t host2device_c_coo(alphasparse_matrix_t A);
alphasparse_status_t host2device_z_coo(alphasparse_matrix_t A);

alphasparse_status_t host2device_s_ell(alphasparse_matrix_t A);
alphasparse_status_t host2device_d_ell(alphasparse_matrix_t A);
alphasparse_status_t host2device_c_ell(alphasparse_matrix_t A);
alphasparse_status_t host2device_z_ell(alphasparse_matrix_t A);

alphasparse_status_t host2device_s_bsr(alphasparse_matrix_t A);
alphasparse_status_t host2device_d_bsr(alphasparse_matrix_t A);
alphasparse_status_t host2device_c_bsr(alphasparse_matrix_t A);
alphasparse_status_t host2device_z_bsr(alphasparse_matrix_t A);

alphasparse_status_t host2device_s_gebsr(alphasparse_matrix_t A);
alphasparse_status_t host2device_d_gebsr(alphasparse_matrix_t A);
alphasparse_status_t host2device_c_gebsr(alphasparse_matrix_t A);
alphasparse_status_t host2device_z_gebsr(alphasparse_matrix_t A);

alphasparse_status_t host2device_s_hyb(alphasparse_matrix_t A);
alphasparse_status_t host2device_d_hyb(alphasparse_matrix_t A);
alphasparse_status_t host2device_c_hyb(alphasparse_matrix_t A);
alphasparse_status_t host2device_z_hyb(alphasparse_matrix_t A);

/**
 * general format for spmv
 * 
 */
alphasparse_status_t alphasparse_s_mv_dcu(alphasparse_dcu_handle_t handle,
                                          const alphasparse_operation_t operation,
                                          const float *alpha,
                                          const alphasparse_matrix_t A,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const float *x,
                                          const float *beta,
                                          float *y);

alphasparse_status_t alphasparse_d_mv_dcu(alphasparse_dcu_handle_t handle,
                                          const alphasparse_operation_t operation,
                                          const double *alpha,
                                          const alphasparse_matrix_t A,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const double *x,
                                          const double *beta,
                                          double *y);

alphasparse_status_t alphasparse_c_mv_dcu(alphasparse_dcu_handle_t handle,
                                          const alphasparse_operation_t operation,
                                          const ALPHA_Complex8 *alpha,
                                          const alphasparse_matrix_t A,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const ALPHA_Complex8 *x,
                                          const ALPHA_Complex8 *beta,
                                          ALPHA_Complex8 *y);

alphasparse_status_t alphasparse_z_mv_dcu(alphasparse_dcu_handle_t handle,
                                          const alphasparse_operation_t operation,
                                          const ALPHA_Complex16 *alpha,
                                          const alphasparse_matrix_t A,
                                          const alpha_dcu_matrix_descr_t descr,
                                          const ALPHA_Complex16 *x,
                                          const ALPHA_Complex16 *beta,
                                          ALPHA_Complex16 *y);

#ifdef __cplusplus
}
#endif