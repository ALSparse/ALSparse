#pragma once

#include "../spmat.h"

alphasparse_status_t dcu_add_d_csr(const spmat_csr_d_t *A, const double alpha, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t dcu_add_d_csr_trans(const spmat_csr_d_t *A, const double alpha, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t dcu_add_d_csr_conj(const spmat_csr_d_t *A, const double alpha, const spmat_csr_d_t *B, spmat_csr_d_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t dcu_gemv_d_csr(alphasparse_dcu_handle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT nnz,
                                   const double alpha,
                                   const double *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   alphasparse_dcu_mat_info_t info,
                                   const double *x,
                                   const double beta,
                                   double *y);
// alpha*A^T*x + beta*y
alphasparse_status_t dcu_gemv_d_csr_trans(alphasparse_dcu_handle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT nnz,
                                   const double alpha,
                                   const double *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   alphasparse_dcu_mat_info_t info,
                                   const double *x,
                                   const double beta,
                                   double *y);
// alpha*(L+D+L')*x + beta*y
alphasparse_status_t dcu_symv_d_csr_n_lo(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t dcu_symv_d_csr_u_lo(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t dcu_symv_d_csr_n_hi(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t dcu_symv_d_csr_u_hi(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_n_lo(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_u_lo(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_n_hi(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_u_hi(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_n_lo_trans(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_u_lo_trans(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_n_hi_trans(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_u_hi_trans(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_n_lo_conj(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_u_lo_conj(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_n_hi_conj(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_csr_u_hi_conj(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*D*x + beta*y
alphasparse_status_t dcu_diagmv_d_csr_n(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*x + beta*y
alphasparse_status_t dcu_diagmv_d_csr_u(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t dcu_gemm_d_csr_row(alphasparse_dcu_handle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT k,
                                   ALPHA_INT nnz,
                                   double alpha,
                                   const double *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   const double *B,
                                   ALPHA_INT ldb,
                                   double beta,
                                   double *C,
                                   ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparse_status_t dcu_gemm_d_csr_row_transA(alphasparse_dcu_handle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          double alpha,
                                          const double *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const double *B,
                                          ALPHA_INT ldb,
                                          double beta,
                                          double *C,
                                          ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparse_status_t dcu_gemm_d_csr_row_transB(alphasparse_dcu_handle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          double alpha,
                                          const double *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const double *B,
                                          ALPHA_INT ldb,
                                          double beta,
                                          double *C,
                                          ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparse_status_t dcu_gemm_d_csr_row_transAB(alphasparse_dcu_handle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           double alpha,
                                           const double *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const double *B,
                                           ALPHA_INT ldb,
                                           double beta,
                                           double *C,
                                           ALPHA_INT ldc);

// alpha*A*B + beta*C
alphasparse_status_t dcu_gemm_d_csr_col(alphasparse_dcu_handle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT k,
                                   ALPHA_INT nnz,
                                   double alpha,
                                   const double *csr_val,
                                   const ALPHA_INT *csr_row_ptr,
                                   const ALPHA_INT *csr_col_ind,
                                   const double *B,
                                   ALPHA_INT ldb,
                                   double beta,
                                   double *C,
                                   ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparse_status_t dcu_gemm_d_csr_col_transA(alphasparse_dcu_handle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          double alpha,
                                          const double *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const double *B,
                                          ALPHA_INT ldb,
                                          double beta,
                                          double *C,
                                          ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparse_status_t dcu_gemm_d_csr_col_transB(alphasparse_dcu_handle_t handle,
                                          ALPHA_INT m,
                                          ALPHA_INT n,
                                          ALPHA_INT k,
                                          ALPHA_INT nnz,
                                          double alpha,
                                          const double *csr_val,
                                          const ALPHA_INT *csr_row_ptr,
                                          const ALPHA_INT *csr_col_ind,
                                          const double *B,
                                          ALPHA_INT ldb,
                                          double beta,
                                          double *C,
                                          ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparse_status_t dcu_gemm_d_csr_col_transAB(alphasparse_dcu_handle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           double alpha,
                                           const double *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const double *B,
                                           ALPHA_INT ldb,
                                           double beta,
                                           double *C,
                                           ALPHA_INT ldc);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_d_csr_n_lo_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_d_csr_u_lo_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_d_csr_n_hi_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_d_csr_u_hi_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_d_csr_n_lo_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_d_csr_u_lo_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_d_csr_n_hi_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_d_csr_u_hi_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_lo_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_lo_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_hi_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_hi_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_lo_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_lo_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_hi_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_hi_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_lo_row_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_lo_row_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_hi_row_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_hi_row_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_lo_col_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_lo_col_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_hi_col_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_hi_col_trans(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_lo_row_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_lo_row_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_hi_row_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_hi_row_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_lo_col_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_lo_col_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_n_hi_col_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_csr_u_hi_col_conj(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t dcu_diagmm_d_csr_n_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t dcu_diagmm_d_csr_u_row(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t dcu_diagmm_d_csr_n_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t dcu_diagmm_d_csr_u_col(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t dcu_gemmi_d_csr(alphasparse_dcu_handle_t handle,
                                    ALPHA_INT m,
                                    ALPHA_INT n,
                                    ALPHA_INT k,
                                    ALPHA_INT nnz,
                                    const double alpha,
                                    const double *A,
                                    ALPHA_INT lda,
                                    const double *csr_val,
                                    const ALPHA_INT *csr_row_ptr,
                                    const ALPHA_INT *csr_col_ind,
                                    const double beta,
                                    double *C,
                                    ALPHA_INT ldc);
// alpha*A^T*B + beta*C
alphasparse_status_t dcu_gemmi_d_csr_transA(alphasparse_dcu_handle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           const double alpha,
                                           const double *A,
                                           ALPHA_INT lda,
                                           const double *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const double beta,
                                           double *C,
                                           ALPHA_INT ldc);

// alpha*A*B^T + beta*C
alphasparse_status_t dcu_gemmi_d_csr_transB(alphasparse_dcu_handle_t handle,
                                           ALPHA_INT m,
                                           ALPHA_INT n,
                                           ALPHA_INT k,
                                           ALPHA_INT nnz,
                                           const double alpha,
                                           const double *A,
                                           ALPHA_INT lda,
                                           const double *csr_val,
                                           const ALPHA_INT *csr_row_ptr,
                                           const ALPHA_INT *csr_col_ind,
                                           const double beta,
                                           double *C,
                                           ALPHA_INT ldc);

// alpha*A^T*B^T + beta*C
alphasparse_status_t dcu_gemmi_d_csr_transAB(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT m,
                                            ALPHA_INT n,
                                            ALPHA_INT k,
                                            ALPHA_INT nnz,
                                            const double alpha,
                                            const double *A,
                                            ALPHA_INT lda,
                                            const double *csr_val,
                                            const ALPHA_INT *csr_row_ptr,
                                            const ALPHA_INT *csr_col_ind,
                                            const double beta,
                                            double *C,
                                            ALPHA_INT ldc);

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t dcu_spmmd_d_csr_row(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t dcu_spmmd_d_csr_col(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_d_csr_row_trans(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_d_csr_col_trans(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);

// A^T*B
alphasparse_status_t dcu_spmmd_d_csr_row_conj(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_d_csr_col_conj(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);

alphasparse_status_t dcu_spmm_d_csr(const spmat_csr_d_t *A, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t dcu_spmm_d_csr_trans(const spmat_csr_d_t *A, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t dcu_spmm_d_csr_conj(const spmat_csr_d_t *A, const spmat_csr_d_t *B, spmat_csr_d_t **C);

// -----------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------
// C = alpha * A * B + beta * D
alphasparse_status_t dcu_spgemm_d_csr(alphasparse_dcu_handle_t handle,
                                     ALPHA_INT m,
                                     ALPHA_INT n,
                                     ALPHA_INT k,
                                     const double alpha,
                                     ALPHA_INT nnz_A,
                                     const double *csr_val_A,
                                     const ALPHA_INT *csr_row_ptr_A,
                                     const ALPHA_INT *csr_col_ind_A,
                                     ALPHA_INT nnz_B,
                                     const double *csr_val_B,
                                     const ALPHA_INT *csr_row_ptr_B,
                                     const ALPHA_INT *csr_col_ind_B,
                                     const double beta,
                                     ALPHA_INT nnz_D,
                                     const double *csr_val_D,
                                     const ALPHA_INT *csr_row_ptr_D,
                                     const ALPHA_INT *csr_col_ind_D,
                                     double *csr_val_C,
                                     const ALPHA_INT *csr_row_ptr_C,
                                     ALPHA_INT *csr_col_ind_C,
                                     const alphasparse_dcu_mat_info_t info_C,
                                     void *temp_buffer);

alphasparse_status_t dcu_spgemm_d_csr_transA(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT m,
                                            ALPHA_INT n,
                                            ALPHA_INT k,
                                            const double alpha,
                                            ALPHA_INT nnz_A,
                                            const double *csr_val_A,
                                            const ALPHA_INT *csr_row_ptr_A,
                                            const ALPHA_INT *csr_col_ind_A,
                                            ALPHA_INT nnz_B,
                                            const double *csr_val_B,
                                            const ALPHA_INT *csr_row_ptr_B,
                                            const ALPHA_INT *csr_col_ind_B,
                                            const double beta,
                                            ALPHA_INT nnz_D,
                                            const double *csr_val_D,
                                            const ALPHA_INT *csr_row_ptr_D,
                                            const ALPHA_INT *csr_col_ind_D,
                                            double *csr_val_C,
                                            const ALPHA_INT *csr_row_ptr_C,
                                            ALPHA_INT *csr_col_ind_C,
                                            const alphasparse_dcu_mat_info_t info_C,
                                            void *temp_buffer);

alphasparse_status_t dcu_spgemm_d_csr_transB(alphasparse_dcu_handle_t handle,
                                            ALPHA_INT m,
                                            ALPHA_INT n,
                                            ALPHA_INT k,
                                            const double alpha,
                                            ALPHA_INT nnz_A,
                                            const double *csr_val_A,
                                            const ALPHA_INT *csr_row_ptr_A,
                                            const ALPHA_INT *csr_col_ind_A,
                                            ALPHA_INT nnz_B,
                                            const double *csr_val_B,
                                            const ALPHA_INT *csr_row_ptr_B,
                                            const ALPHA_INT *csr_col_ind_B,
                                            const double beta,
                                            ALPHA_INT nnz_D,
                                            const double *csr_val_D,
                                            const ALPHA_INT *csr_row_ptr_D,
                                            const ALPHA_INT *csr_col_ind_D,
                                            double *csr_val_C,
                                            const ALPHA_INT *csr_row_ptr_C,
                                            ALPHA_INT *csr_col_ind_C,
                                            const alphasparse_dcu_mat_info_t info_C,
                                            void *temp_buffer);

alphasparse_status_t dcu_spgemm_d_csr_transAB(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT n,
                                             ALPHA_INT k,
                                             const double alpha,
                                             ALPHA_INT nnz_A,
                                             const double *csr_val_A,
                                             const ALPHA_INT *csr_row_ptr_A,
                                             const ALPHA_INT *csr_col_ind_A,
                                             ALPHA_INT nnz_B,
                                             const double *csr_val_B,
                                             const ALPHA_INT *csr_row_ptr_B,
                                             const ALPHA_INT *csr_col_ind_B,
                                             const double beta,
                                             ALPHA_INT nnz_D,
                                             const double *csr_val_D,
                                             const ALPHA_INT *csr_row_ptr_D,
                                             const ALPHA_INT *csr_col_ind_D,
                                             double *csr_val_C,
                                             const ALPHA_INT *csr_row_ptr_C,
                                             ALPHA_INT *csr_col_ind_C,
                                             const alphasparse_dcu_mat_info_t info_C,
                                             void *temp_buffer);
// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t dcu_trsv_d_csr_n_lo(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const double *x,
                                        double *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*x
alphasparse_status_t dcu_trsv_d_csr_u_lo(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const double *x,
                                        double *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*x
alphasparse_status_t dcu_trsv_d_csr_n_hi(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const double *x,
                                        double *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*x
alphasparse_status_t dcu_trsv_d_csr_u_hi(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        alphasparse_dcu_mat_info_t info,
                                        const double *x,
                                        double *y,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_csr_n_lo_trans(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const double alpha,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const double *x,
                                              double *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_csr_u_lo_trans(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const double alpha,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const double *x,
                                              double *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_csr_n_hi_trans(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const double alpha,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const double *x,
                                              double *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_csr_u_hi_trans(alphasparse_dcu_handle_t handle,
                                              ALPHA_INT m,
                                              ALPHA_INT nnz,
                                              const double alpha,
                                              const double *csr_val,
                                              const ALPHA_INT *csr_row_ptr,
                                              const ALPHA_INT *csr_col_ind,
                                              alphasparse_dcu_mat_info_t info,
                                              const double *x,
                                              double *y,
                                              alphasparse_dcu_solve_policy_t policy,
                                              void *temp_buffer);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_csr_n_lo_conj(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const double alpha,
                                             const double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const double *x,
                                             double *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_csr_u_lo_conj(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const double alpha,
                                             const double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const double *x,
                                             double *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_csr_n_hi_conj(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const double alpha,
                                             const double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const double *x,
                                             double *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_csr_u_hi_conj(alphasparse_dcu_handle_t handle,
                                             ALPHA_INT m,
                                             ALPHA_INT nnz,
                                             const double alpha,
                                             const double *csr_val,
                                             const ALPHA_INT *csr_row_ptr,
                                             const ALPHA_INT *csr_col_ind,
                                             alphasparse_dcu_mat_info_t info,
                                             const double *x,
                                             double *y,
                                             alphasparse_dcu_solve_policy_t policy,
                                             void *temp_buffer);

// alpha*inv(D)*x
alphasparse_status_t dcu_diagsv_d_csr_n(alphasparse_dcu_handle_t handle,
                                       ALPHA_INT m,
                                       ALPHA_INT nnz,
                                       const double alpha,
                                       const double *csr_val,
                                       const ALPHA_INT *csr_row_ptr,
                                       const ALPHA_INT *csr_col_ind,
                                       alphasparse_dcu_mat_info_t info,
                                       const double *x,
                                       double *y,
                                       alphasparse_dcu_solve_policy_t policy,
                                       void *temp_buffer);
// alpha*x
alphasparse_status_t dcu_diagsv_d_csr_u(alphasparse_dcu_handle_t handle,
                                       ALPHA_INT m,
                                       ALPHA_INT nnz,
                                       const double alpha,
                                       const double *csr_val,
                                       const ALPHA_INT *csr_row_ptr,
                                       const ALPHA_INT *csr_col_ind,
                                       alphasparse_dcu_mat_info_t info,
                                       const double *x,
                                       double *y,
                                       alphasparse_dcu_solve_policy_t policy,
                                       void *temp_buffer);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi(alphasparse_dcu_handle_t handle,
                                        ALPHA_INT m,
                                        ALPHA_INT nrhs,
                                        ALPHA_INT nnz,
                                        const double alpha,
                                        const double *csr_val,
                                        const ALPHA_INT *csr_row_ptr,
                                        const ALPHA_INT *csr_col_ind,
                                        double *B,
                                        ALPHA_INT ldb,
                                        alphasparse_dcu_mat_info_t info,
                                        alphasparse_dcu_solve_policy_t policy,
                                        void *temp_buffer);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi_transA(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi_transB(alphasparse_dcu_handle_t handle,
                                               ALPHA_INT m,
                                               ALPHA_INT nrhs,
                                               ALPHA_INT nnz,
                                               const double alpha,
                                               const double *csr_val,
                                               const ALPHA_INT *csr_row_ptr,
                                               const ALPHA_INT *csr_col_ind,
                                               double *B,
                                               ALPHA_INT ldb,
                                               alphasparse_dcu_mat_info_t info,
                                               alphasparse_dcu_solve_policy_t policy,
                                               void *temp_buffer);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_n_lo_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_csr_u_lo_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_n_hi_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_csr_u_hi_transAB(alphasparse_dcu_handle_t handle,
                                                ALPHA_INT m,
                                                ALPHA_INT nrhs,
                                                ALPHA_INT nnz,
                                                const double alpha,
                                                const double *csr_val,
                                                const ALPHA_INT *csr_row_ptr,
                                                const ALPHA_INT *csr_col_ind,
                                                double *B,
                                                ALPHA_INT ldb,
                                                alphasparse_dcu_mat_info_t info,
                                                alphasparse_dcu_solve_policy_t policy,
                                                void *temp_buffer);
// alpha*inv(D)*x
alphasparse_status_t dcu_diagsm_d_csr_n_row(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t dcu_diagsm_d_csr_u_row(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t dcu_diagsm_d_csr_n_col(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t dcu_diagsm_d_csr_u_col(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

alphasparse_status_t dcu_geam_d_csr(alphasparse_dcu_handle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   const double alpha,
                                   ALPHA_INT nnz_A,
                                   const double *csr_val_A,
                                   const ALPHA_INT *csr_row_ptr_A,
                                   const ALPHA_INT *csr_col_ind_A,
                                   const double beta,
                                   ALPHA_INT nnz_B,
                                   const double *csr_val_B,
                                   const ALPHA_INT *csr_row_ptr_B,
                                   const ALPHA_INT *csr_col_ind_B,
                                   double *csr_val_C,
                                   const ALPHA_INT *csr_row_ptr_C,
                                   ALPHA_INT *csr_col_ind_C);
                                   
alphasparse_status_t dcu_set_value_d_csr(spmat_csr_d_t *A, const ALPHA_INT row, const ALPHA_INT col, const double value);