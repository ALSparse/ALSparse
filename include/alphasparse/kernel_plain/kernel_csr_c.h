#pragma once

#include "../spmat.h"

alphasparse_status_t add_c_csr_plain(const spmat_csr_c_t *A, const ALPHA_Complex8 alpha, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparse_status_t add_c_csr_trans_plain(const spmat_csr_c_t *A, const ALPHA_Complex8 alpha, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparse_status_t add_c_csr_conj_plain(const spmat_csr_c_t *A, const ALPHA_Complex8 alpha, const spmat_csr_c_t *B, spmat_csr_c_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t gemv_c_csr_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_c_csr_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_c_csr_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_c_csr_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_c_csr_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_c_csr_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_c_csr_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_c_csr_n_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_c_csr_u_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_c_csr_n_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_c_csr_u_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t hermv_c_csr_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t hermv_c_csr_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t hermv_c_csr_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t hermv_c_csr_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+D+L')^T*x + beta*y
alphasparse_status_t hermv_c_csr_n_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I+L')^T*x + beta*y
alphasparse_status_t hermv_c_csr_u_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+D+U)^T*x + beta*y
alphasparse_status_t hermv_c_csr_n_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U'+I+U)^T*x + beta*y
alphasparse_status_t hermv_c_csr_u_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t trmv_c_csr_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_c_csr_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_c_csr_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_c_csr_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_c_csr_n_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_c_csr_u_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_c_csr_n_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_c_csr_u_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_c_csr_n_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_c_csr_u_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_c_csr_n_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_c_csr_u_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// alpha*D*x + beta*y
alphasparse_status_t diagmv_c_csr_n_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);
// alpha*x + beta*y
alphasparse_status_t diagmv_c_csr_u_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_Complex8 beta, ALPHA_Complex8 *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t gemm_c_csr_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_c_csr_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_c_csr_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_c_csr_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_c_csr_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_c_csr_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_csr_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_csr_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_csr_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_csr_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_csr_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_csr_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_csr_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_csr_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);


// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_csr_n_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_csr_u_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_csr_n_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_csr_u_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_c_csr_n_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_c_csr_u_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_c_csr_n_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_c_csr_u_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t hermm_c_csr_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t hermm_c_csr_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t hermm_c_csr_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t hermm_c_csr_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t hermm_c_csr_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t hermm_c_csr_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t hermm_c_csr_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t hermm_c_csr_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparse_status_t hermm_c_csr_n_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparse_status_t hermm_c_csr_u_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparse_status_t hermm_c_csr_n_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparse_status_t hermm_c_csr_u_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T^T*B + beta*C
alphasparse_status_t hermm_c_csr_n_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I+L')^T*B + beta*C
alphasparse_status_t hermm_c_csr_u_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)^T*B + beta*C
alphasparse_status_t hermm_c_csr_n_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)^T*B + beta*C
alphasparse_status_t hermm_c_csr_u_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_c_csr_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_c_csr_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t trmm_c_csr_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t trmm_c_csr_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_c_csr_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_c_csr_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t trmm_c_csr_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t trmm_c_csr_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_c_csr_n_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_c_csr_u_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t diagmm_c_csr_n_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_c_csr_u_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t diagmm_c_csr_n_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_c_csr_u_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *mat, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex8 beta, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t spmmd_c_csr_row_plain(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t spmmd_c_csr_col_plain(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_c_csr_row_trans_plain(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_c_csr_col_trans_plain(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

// A^T*B
alphasparse_status_t spmmd_c_csr_row_conj_plain(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_c_csr_col_conj_plain(const spmat_csr_c_t *matA, const spmat_csr_c_t *matB, ALPHA_Complex8 *C, const ALPHA_INT ldc);

alphasparse_status_t spmm_c_csr_plain(const spmat_csr_c_t *A, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparse_status_t spmm_c_csr_trans_plain(const spmat_csr_c_t *A, const spmat_csr_c_t *B, spmat_csr_c_t **C);
alphasparse_status_t spmm_c_csr_conj_plain(const spmat_csr_c_t *A, const spmat_csr_c_t *B, spmat_csr_c_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t trsv_c_csr_n_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L)*x
alphasparse_status_t trsv_c_csr_u_lo_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_c_csr_n_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_c_csr_u_hi_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_csr_n_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_csr_u_lo_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_csr_n_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_csr_u_hi_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_csr_n_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_c_csr_u_lo_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_csr_n_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_c_csr_u_hi_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(D)*x
alphasparse_status_t diagsv_c_csr_n_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);
// alpha*x
alphasparse_status_t diagsv_c_csr_u_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, ALPHA_Complex8 *y);

// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_n_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_u_lo_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_n_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_u_hi_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_n_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_u_lo_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_n_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_u_hi_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_n_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_u_lo_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_n_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_u_hi_row_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_n_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_u_lo_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_n_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_u_hi_col_trans_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_n_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_u_lo_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_n_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_u_hi_row_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_n_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_c_csr_u_lo_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_n_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_c_csr_u_hi_col_conj_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t diagsm_c_csr_n_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_c_csr_u_row_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t diagsm_c_csr_n_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_c_csr_u_col_plain(const ALPHA_Complex8 alpha, const spmat_csr_c_t *A, const ALPHA_Complex8 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex8 *y, const ALPHA_INT ldy);

alphasparse_status_t set_value_c_csr_plain (spmat_csr_c_t * A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex8 value);