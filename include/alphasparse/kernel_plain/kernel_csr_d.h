#pragma once

#include "../spmat.h"

alphasparse_status_t add_d_csr_plain(const spmat_csr_d_t *A, const double alpha, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t add_d_csr_trans_plain(const spmat_csr_d_t *A, const double alpha, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t add_d_csr_conj_plain(const spmat_csr_d_t *A, const double alpha, const spmat_csr_d_t *B, spmat_csr_d_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t gemv_d_csr_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_d_csr_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_d_csr_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_d_csr_n_lo_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_d_csr_u_lo_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_d_csr_n_hi_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_d_csr_u_hi_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t trmv_d_csr_n_lo_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_d_csr_u_lo_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_d_csr_n_hi_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_d_csr_u_hi_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_d_csr_n_lo_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_d_csr_u_lo_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_d_csr_n_hi_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_d_csr_u_hi_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_d_csr_n_lo_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_d_csr_u_lo_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_d_csr_n_hi_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_d_csr_u_hi_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// alpha*D*x + beta*y
alphasparse_status_t diagmv_d_csr_n_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);
// alpha*x + beta*y
alphasparse_status_t diagmv_d_csr_u_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const double beta, double *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t gemm_d_csr_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_d_csr_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_d_csr_row_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_d_csr_col_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_d_csr_row_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_d_csr_col_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_d_csr_n_lo_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_d_csr_u_lo_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_d_csr_n_hi_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_d_csr_u_hi_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_d_csr_n_lo_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_d_csr_u_lo_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_d_csr_n_hi_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_d_csr_u_hi_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_d_csr_n_lo_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_d_csr_u_lo_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t trmm_d_csr_n_hi_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t trmm_d_csr_u_hi_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_d_csr_n_lo_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_d_csr_u_lo_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t trmm_d_csr_n_hi_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t trmm_d_csr_u_hi_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_lo_row_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_lo_row_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_hi_row_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_hi_row_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_lo_col_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_lo_col_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_hi_col_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_hi_col_trans_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_lo_row_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_lo_row_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_hi_row_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_hi_row_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_lo_col_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_lo_col_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_d_csr_n_hi_col_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_d_csr_u_hi_col_conj_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t diagmm_d_csr_n_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_d_csr_u_row_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t diagmm_d_csr_n_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_d_csr_u_col_plain(const double alpha, const spmat_csr_d_t *mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t spmmd_d_csr_row_plain(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t spmmd_d_csr_col_plain(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_d_csr_row_trans_plain(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_d_csr_col_trans_plain(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);

// A^T*B
alphasparse_status_t spmmd_d_csr_row_conj_plain(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_d_csr_col_conj_plain(const spmat_csr_d_t *matA, const spmat_csr_d_t *matB, double *C, const ALPHA_INT ldc);

alphasparse_status_t spmm_d_csr_plain(const spmat_csr_d_t *A, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t spmm_d_csr_trans_plain(const spmat_csr_d_t *A, const spmat_csr_d_t *B, spmat_csr_d_t **C);
alphasparse_status_t spmm_d_csr_conj_plain(const spmat_csr_d_t *A, const spmat_csr_d_t *B, spmat_csr_d_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t trsv_d_csr_n_lo_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(L)*x
alphasparse_status_t trsv_d_csr_u_lo_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_d_csr_n_hi_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_d_csr_u_hi_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_d_csr_n_lo_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_d_csr_u_lo_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_d_csr_n_hi_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_d_csr_u_hi_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_d_csr_n_lo_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_d_csr_u_lo_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_d_csr_n_hi_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_d_csr_u_hi_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);

// alpha*inv(D)*x
alphasparse_status_t diagsv_d_csr_n_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);
// alpha*x
alphasparse_status_t diagsv_d_csr_u_plain(const double alpha, const spmat_csr_d_t *A, const double *x, double *y);

// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_n_lo_row_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_u_lo_row_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_n_hi_row_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_u_hi_row_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_n_lo_col_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_u_lo_col_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_n_hi_col_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_u_hi_col_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_n_lo_row_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_u_lo_row_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_n_hi_row_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_u_hi_row_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_n_lo_col_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_u_lo_col_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_n_hi_col_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_u_hi_col_trans_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_n_lo_row_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_u_lo_row_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_n_hi_row_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_u_hi_row_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_n_lo_col_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_d_csr_u_lo_col_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_n_hi_col_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_d_csr_u_hi_col_conj_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t diagsm_d_csr_n_row_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_d_csr_u_row_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t diagsm_d_csr_n_col_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_d_csr_u_col_plain(const double alpha, const spmat_csr_d_t *A, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, double *y, const ALPHA_INT ldy);

alphasparse_status_t set_value_d_csr_plain (spmat_csr_d_t * A, const ALPHA_INT row, const ALPHA_INT col, const double value);