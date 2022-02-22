#pragma once

#include "../spmat.h"

alphasparse_status_t add_c_bsr(const spmat_bsr_c_t *A, const ALPHA_Complex8 alpha, const spmat_bsr_c_t *B, spmat_bsr_c_t **C);
alphasparse_status_t add_c_bsr_trans(const spmat_bsr_c_t *A, const ALPHA_Complex8 alpha, const spmat_bsr_c_t *B, spmat_bsr_c_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t gemv_c_bsr(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_c_bsr_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_c_bsr_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_c_bsr_n_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_c_bsr_u_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_c_bsr_n_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_c_bsr_u_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);


// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_c_bsr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_c_bsr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_c_bsr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_c_bsr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t hermv_c_bsr_n_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t hermv_c_bsr_u_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t hermv_c_bsr_n_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t hermv_c_bsr_u_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t hermv_c_bsr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t hermv_c_bsr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t hermv_c_bsr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t hermv_c_bsr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t trmv_c_bsr_n_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_c_bsr_u_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_c_bsr_n_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_c_bsr_u_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_c_bsr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_c_bsr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_c_bsr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_c_bsr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

alphasparse_status_t trmv_c_bsr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_c_bsr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_c_bsr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_c_bsr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);


// alpha*D*x + beta*y
alphasparse_status_t diagmv_c_bsr_n(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*x + beta*y
alphasparse_status_t diagmv_c_bsr_u(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t gemm_c_bsr_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_c_bsr_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_c_bsr_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_c_bsr_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

alphasparse_status_t gemm_c_bsr_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_c_bsr_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_bsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_bsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_bsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_bsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_bsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_bsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_bsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_bsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);


// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_bsr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_bsr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_bsr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_bsr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_bsr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_bsr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_bsr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_bsr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t hermm_c_bsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t hermm_c_bsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t hermm_c_bsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t hermm_c_bsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t hermm_c_bsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t hermm_c_bsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t hermm_c_bsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t hermm_c_bsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t hermm_c_bsr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_c_bsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_c_bsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t trmm_c_bsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t trmm_c_bsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_c_bsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_c_bsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t trmm_c_bsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t trmm_c_bsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);


// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_bsr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t diagmm_c_bsr_n_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_c_bsr_u_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t diagmm_c_bsr_n_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_c_bsr_u_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t spmmd_c_bsr_row(const spmat_bsr_c_t *matA, const spmat_bsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t spmmd_c_bsr_col(const spmat_bsr_c_t *matA, const spmat_bsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_c_bsr_row_trans(const spmat_bsr_c_t *matA, const spmat_bsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_c_bsr_col_trans(const spmat_bsr_c_t *matA, const spmat_bsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_c_bsr_row_conj(const spmat_bsr_c_t *matA, const spmat_bsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_c_bsr_col_conj(const spmat_bsr_c_t *matA, const spmat_bsr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

alphasparse_status_t spmm_c_bsr(const spmat_bsr_c_t *A, const spmat_bsr_c_t *B, spmat_bsr_c_t **C);
alphasparse_status_t spmm_c_bsr_trans(const spmat_bsr_c_t *A, const spmat_bsr_c_t *B, spmat_bsr_c_t **C);
alphasparse_status_t spmm_c_bsr_conj(const spmat_bsr_c_t *A, const spmat_bsr_c_t *B, spmat_bsr_c_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t trsv_c_bsr_n_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L)*x
alphasparse_status_t trsv_c_bsr_u_lo(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_c_bsr_n_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_c_bsr_u_hi(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_bsr_n_lo_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_bsr_u_lo_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_bsr_n_hi_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_bsr_u_hi_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_bsr_n_lo_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_bsr_u_lo_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_bsr_n_hi_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_bsr_u_hi_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(D)*x
alphasparse_status_t diagsv_c_bsr_n(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*x
alphasparse_status_t diagsv_c_bsr_u(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_n_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_u_lo_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_n_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_u_hi_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_n_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_u_lo_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_n_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_u_hi_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_n_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_u_lo_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_n_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_u_hi_row_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_n_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_u_lo_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_n_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_u_hi_col_trans(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_n_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_u_lo_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_n_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_u_hi_row_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_n_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_bsr_u_lo_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_n_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_bsr_u_hi_col_conj(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t diagsm_c_bsr_n_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_c_bsr_u_row(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t diagsm_c_bsr_n_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_c_bsr_u_col(const ALPHA_Complex8 alpha, const spmat_bsr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

alphasparse_status_t axpy_c(const ALPHA_INT nz,  const ALPHA_Complex8 a,  const ALPHA_Complex8* x,  const ALPHA_INT* indx,  ALPHA_Complex8* y);

alphasparse_status_t gthr_c(const ALPHA_INT nz,	const ALPHA_Complex8* y, ALPHA_Complex8* x, const ALPHA_INT* indx);

alphasparse_status_t rot_c(const ALPHA_INT nz, ALPHA_Complex8* x, const ALPHA_INT* indx, ALPHA_Complex8* y, const ALPHA_Complex8 c, const ALPHA_Complex8 s);

alphasparse_status_t sctr_c(const ALPHA_INT nz, const ALPHA_Complex8* x, const ALPHA_INT* indx, ALPHA_Complex8* y);

ALPHA_Complex8 doti_c(const ALPHA_INT nz,  const ALPHA_Complex8* x,  const ALPHA_INT* indx, const ALPHA_Complex8* y);

alphasparse_status_t set_value_c_bsr (spmat_bsr_c_t * A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex8 value);
alphasparse_status_t update_values_c_bsr (spmat_bsr_c_t * A, const ALPHA_INT nvalues, const ALPHA_INT *indx, const ALPHA_INT *indy, ALPHA_Complex8 *values);


