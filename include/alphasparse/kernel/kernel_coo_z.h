#pragma once

#include "../spmat.h"

alphasparse_status_t add_z_coo(const spmat_coo_z_t *A, const ALPHA_Complex16 alpha, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparse_status_t add_z_coo_trans(const spmat_coo_z_t *A, const ALPHA_Complex16 alpha, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparse_status_t add_z_coo_conj(const spmat_coo_z_t *A, const ALPHA_Complex16 alpha, const spmat_coo_z_t *B, spmat_coo_z_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t gemv_z_coo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_z_coo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^H*x + beta*y
alphasparse_status_t gemv_z_coo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_z_coo_n_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_z_coo_u_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_z_coo_n_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_z_coo_u_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_z_coo_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_z_coo_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_z_coo_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_z_coo_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t trmv_z_coo_n_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_z_coo_u_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_z_coo_n_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_z_coo_u_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_z_coo_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_z_coo_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_z_coo_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_z_coo_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^H*x + beta*y
alphasparse_status_t trmv_z_coo_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^H*x + beta*y
alphasparse_status_t trmv_z_coo_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^H*x + beta*y
alphasparse_status_t trmv_z_coo_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^H*x + beta*y
alphasparse_status_t trmv_z_coo_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*D*x + beta*y
alphasparse_status_t diagmv_z_coo_n(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*x + beta*y
alphasparse_status_t diagmv_z_coo_u(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t gemm_z_coo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_z_coo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_z_coo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_z_coo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*A^H*B + beta*C
alphasparse_status_t gemm_z_coo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_z_coo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_coo_n_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_z_coo_u_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_coo_n_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_coo_u_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_coo_n_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_z_coo_u_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_coo_n_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_coo_u_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_coo_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_z_coo_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_coo_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_coo_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_coo_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_z_coo_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_coo_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_coo_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_z_coo_n_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_z_coo_u_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t trmm_z_coo_n_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t trmm_z_coo_u_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_z_coo_n_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_z_coo_u_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t trmm_z_coo_n_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t trmm_z_coo_u_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_coo_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_coo_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t diagmm_z_coo_n_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_z_coo_u_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t diagmm_z_coo_n_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_z_coo_u_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t spmmd_z_coo_row(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t spmmd_z_coo_col(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_coo_row_trans(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_coo_col_trans(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_coo_row_conj(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_coo_col_conj(const spmat_coo_z_t *matA, const spmat_coo_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);

alphasparse_status_t spmm_z_coo(const spmat_coo_z_t *A, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparse_status_t spmm_z_coo_trans(const spmat_coo_z_t *A, const spmat_coo_z_t *B, spmat_coo_z_t **C);
alphasparse_status_t spmm_z_coo_conj(const spmat_coo_z_t *A, const spmat_coo_z_t *B, spmat_coo_z_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t trsv_z_coo_n_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L)*x
alphasparse_status_t trsv_z_coo_u_lo(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_z_coo_n_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_z_coo_u_hi(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_coo_n_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_coo_u_lo_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_coo_n_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_coo_u_hi_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_coo_n_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_coo_u_lo_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_coo_n_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_coo_u_hi_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(D)*x
alphasparse_status_t diagsv_z_coo_n(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*x
alphasparse_status_t diagsv_z_coo_u(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_n_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_u_lo_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_n_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_u_hi_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_n_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_u_lo_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_n_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_u_hi_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_n_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_u_lo_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_n_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_u_hi_row_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_n_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_u_lo_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_n_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_u_hi_col_trans(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_n_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_u_lo_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_n_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_u_hi_row_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_n_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_coo_u_lo_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_n_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_coo_u_hi_col_conj(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t diagsm_z_coo_n_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_z_coo_u_row(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t diagsm_z_coo_n_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_z_coo_u_col(const ALPHA_Complex16 alpha, const spmat_coo_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

alphasparse_status_t set_value_z_coo (spmat_coo_z_t * A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex16 value);

alphasparse_status_t
hermv_z_coo_u_hi(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);
alphasparse_status_t
hermv_z_coo_u_lo(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);
alphasparse_status_t
hermv_z_coo_n_hi(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);
alphasparse_status_t
hermv_z_coo_n_lo(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);
alphasparse_status_t
hermv_z_coo_u_hi_trans(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);
alphasparse_status_t
hermv_z_coo_u_lo_trans(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);
alphasparse_status_t
hermv_z_coo_n_hi_trans(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);
alphasparse_status_t
hermv_z_coo_n_lo_trans(const ALPHA_Complex16 alpha,
		             const spmat_coo_z_t *A,
		             const ALPHA_Complex16 *x,
		             const ALPHA_Complex16 beta,
		             ALPHA_Complex16 *y);

alphasparse_status_t hermm_z_coo_n_hi_col( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_n_lo_col( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);

alphasparse_status_t hermm_z_coo_u_hi_col( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_u_lo_col( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);


alphasparse_status_t hermm_z_coo_n_hi_col_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_n_lo_col_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);

alphasparse_status_t hermm_z_coo_u_hi_col_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_u_lo_col_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);


alphasparse_status_t hermm_z_coo_n_hi_row( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_n_lo_row( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);

alphasparse_status_t hermm_z_coo_u_hi_row( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_u_lo_row( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);


alphasparse_status_t hermm_z_coo_n_hi_row_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_n_lo_row_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);

alphasparse_status_t hermm_z_coo_u_hi_row_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);
												
alphasparse_status_t hermm_z_coo_u_lo_row_trans( const ALPHA_Complex16 alpha, 
												const spmat_coo_z_t *mat, 
												const ALPHA_Complex16 *x, 
												const ALPHA_INT columns, 
												const ALPHA_INT ldx, 
												const ALPHA_Complex16 beta, 
												ALPHA_Complex16 *y, 
												const ALPHA_INT ldy);