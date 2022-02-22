#pragma once

#include "../spmat.h"

alphasparse_status_t dcu_add_d_coo(const spmat_coo_d_t *A, const double alpha, const spmat_coo_d_t *B, spmat_coo_d_t **C);
alphasparse_status_t dcu_add_d_coo_trans(const spmat_coo_d_t *A, const double alpha, const spmat_coo_d_t *B, spmat_coo_d_t **C);
alphasparse_status_t dcu_add_d_coo_conj(const spmat_coo_d_t *A, const double alpha, const spmat_coo_d_t *B, spmat_coo_d_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t dcu_gemv_d_coo(alphasparse_dcu_handle_t handle,
                                   ALPHA_INT m,
                                   ALPHA_INT n,
                                   ALPHA_INT nnz,
                                   const double alpha,
                                   const double *coo_val,
                                   const ALPHA_INT *coo_row_ind,
                                   const ALPHA_INT *coo_col_ind,
                                   const double *x,
                                   const double beta,
                                   double *y);
// alpha*A^T*x + beta*y
alphasparse_status_t dcu_gemv_d_coo_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*A^T*x + beta*y
alphasparse_status_t dcu_gemv_d_coo_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t dcu_symv_d_coo_n_lo(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t dcu_symv_d_coo_u_lo(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t dcu_symv_d_coo_n_hi(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t dcu_symv_d_coo_u_hi(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_n_lo(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_u_lo(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_n_hi(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_u_hi(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_n_lo_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_u_lo_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_n_hi_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_u_hi_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_n_lo_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_u_lo_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_n_hi_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t dcu_trmv_d_coo_u_hi_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);

// alpha*D*x + beta*y
alphasparse_status_t dcu_diagmv_d_coo_n(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);
// alpha*x + beta*y
alphasparse_status_t dcu_diagmv_d_coo_u(const double alpha, const spmat_coo_d_t *A, const double *x, const double beta, double *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t dcu_gemm_d_coo_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
alphasparse_status_t dcu_gemm_d_coo_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t dcu_gemm_d_coo_row_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
alphasparse_status_t dcu_gemm_d_coo_col_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t dcu_gemm_d_coo_row_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
alphasparse_status_t dcu_gemm_d_coo_col_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_d_coo_n_lo_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_d_coo_u_lo_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_d_coo_n_hi_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_d_coo_u_hi_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t dcu_symm_d_coo_n_lo_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t dcu_symm_d_coo_u_lo_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t dcu_symm_d_coo_n_hi_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t dcu_symm_d_coo_u_hi_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_lo_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_lo_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_hi_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_hi_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_lo_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_lo_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_hi_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_hi_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_lo_row_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_lo_row_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_hi_row_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_hi_row_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_lo_col_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_lo_col_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_hi_col_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_hi_col_trans(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_lo_row_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_lo_row_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_hi_row_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_hi_row_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_lo_col_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_lo_col_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_n_hi_col_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t dcu_trmm_d_coo_u_hi_col_conj(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t dcu_diagmm_d_coo_n_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t dcu_diagmm_d_coo_u_row(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t dcu_diagmm_d_coo_n_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t dcu_diagmm_d_coo_u_col(const double alpha, const spmat_coo_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t dcu_spmmd_d_coo_row(const spmat_coo_d_t *matA, const spmat_coo_d_t *matB, double *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t dcu_spmmd_d_coo_col(const spmat_coo_d_t *matA, const spmat_coo_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_d_coo_row_trans(const spmat_coo_d_t *matA, const spmat_coo_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_d_coo_col_trans(const spmat_coo_d_t *matA, const spmat_coo_d_t *matB, double *C, const ALPHA_INT ldc);

// A^T*B
alphasparse_status_t dcu_spmmd_d_coo_row_conj(const spmat_coo_d_t *matA, const spmat_coo_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t dcu_spmmd_d_coo_col_conj(const spmat_coo_d_t *matA, const spmat_coo_d_t *matB, double *C, const ALPHA_INT ldc);

alphasparse_status_t dcu_spmm_d_coo(const spmat_coo_d_t *A, const spmat_coo_d_t *B, spmat_coo_d_t **C);
alphasparse_status_t dcu_spmm_d_coo_trans(const spmat_coo_d_t *A, const spmat_coo_d_t *B, spmat_coo_d_t **C);
alphasparse_status_t dcu_spmm_d_coo_conj(const spmat_coo_d_t *A, const spmat_coo_d_t *B, spmat_coo_d_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t dcu_trsv_d_coo_n_lo(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(L)*x
alphasparse_status_t dcu_trsv_d_coo_u_lo(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(U)*x
alphasparse_status_t dcu_trsv_d_coo_n_hi(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(U)*x
alphasparse_status_t dcu_trsv_d_coo_u_hi(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_coo_n_lo_trans(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_coo_u_lo_trans(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_coo_n_hi_trans(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_coo_u_hi_trans(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_coo_n_lo_conj(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t dcu_trsv_d_coo_u_lo_conj(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_coo_n_hi_conj(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t dcu_trsv_d_coo_u_hi_conj(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);

// alpha*inv(D)*x
alphasparse_status_t dcu_diagsv_d_coo_n(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);
// alpha*x
alphasparse_status_t dcu_diagsv_d_coo_u(const double alpha, const spmat_coo_d_t *A, const double *x, double *y);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_n_lo_row(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_u_lo_row(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_n_hi_row(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_u_hi_row(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_n_lo_col(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_u_lo_col(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_n_hi_col(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_u_hi_col(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_n_lo_row_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_u_lo_row_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_n_hi_row_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_u_hi_row_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_n_lo_col_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_u_lo_col_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_n_hi_col_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_u_hi_col_trans(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_n_lo_row_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_u_lo_row_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_n_hi_row_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_u_hi_row_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_n_lo_col_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t dcu_trsm_d_coo_u_lo_col_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_n_hi_col_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t dcu_trsm_d_coo_u_hi_col_conj(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t dcu_diagsm_d_coo_n_row(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t dcu_diagsm_d_coo_u_row(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t dcu_diagsm_d_coo_n_col(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t dcu_diagsm_d_coo_u_col(const double alpha, const spmat_coo_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

alphasparse_status_t dcu_set_value_d_coo(spmat_coo_d_t *A, const ALPHA_INT row, const ALPHA_INT col, const double value);