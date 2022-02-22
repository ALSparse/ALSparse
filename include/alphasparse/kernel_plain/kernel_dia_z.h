#pragma once

#include "../spmat.h"

alphasparse_status_t add_z_dia_plain(const spmat_dia_z_t *A, const ALPHA_Complex16 alpha, const spmat_dia_z_t *B, spmat_dia_z_t **C);
alphasparse_status_t add_z_dia_trans_plain(const spmat_dia_z_t *A, const ALPHA_Complex16 alpha, const spmat_dia_z_t *B, spmat_dia_z_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t gemv_z_dia_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_z_dia_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_z_dia_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_z_dia_n_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_z_dia_u_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_z_dia_n_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_z_dia_u_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_z_dia_n_lo_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_z_dia_u_lo_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_z_dia_n_hi_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_z_dia_u_hi_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t hermv_z_dia_n_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t hermv_z_dia_u_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t hermv_z_dia_n_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t hermv_z_dia_u_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t hermv_z_dia_n_lo_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t hermv_z_dia_u_lo_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t hermv_z_dia_n_hi_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t hermv_z_dia_u_hi_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t trmv_z_dia_n_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_z_dia_u_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_z_dia_n_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_z_dia_u_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_z_dia_n_lo_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_z_dia_u_lo_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_z_dia_n_hi_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_z_dia_u_hi_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

alphasparse_status_t trmv_z_dia_n_lo_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_z_dia_u_lo_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_z_dia_n_hi_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_z_dia_u_hi_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);

// alpha*D*x + beta*y
alphasparse_status_t diagmv_z_dia_n_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// alpha*x + beta*y
alphasparse_status_t diagmv_z_dia_u_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_Complex16 beta, ALPHA_Complex16 *y);
// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t gemm_z_dia_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_z_dia_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_z_dia_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_z_dia_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

alphasparse_status_t gemm_z_dia_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_z_dia_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_dia_n_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I+L')*B + beta*C
alphasparse_status_t symm_z_dia_u_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_dia_n_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_dia_u_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_dia_n_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I+L')*B + beta*C
alphasparse_status_t symm_z_dia_u_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_dia_n_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_dia_u_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);


// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_dia_n_lo_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I+L')*B + beta*C
alphasparse_status_t symm_z_dia_u_lo_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_dia_n_hi_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_dia_u_hi_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_z_dia_n_lo_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I+L')*B + beta*C
alphasparse_status_t symm_z_dia_u_lo_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+D+U)*B + beta*C
alphasparse_status_t symm_z_dia_n_hi_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U'+I+U)*B + beta*C
alphasparse_status_t symm_z_dia_u_hi_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*_plain(L+D)*B + beta*C
alphasparse_status_t hermm_z_dia_n_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)*B + beta*C
alphasparse_status_t hermm_z_dia_u_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op_plain(U+D)*B + beta*C
alphasparse_status_t hermm_z_dia_n_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op_plain(U+I)*B + beta*C
alphasparse_status_t hermm_z_dia_u_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*_plain(L+D)*B + beta*C
alphasparse_status_t hermm_z_dia_n_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)*B + beta*C
alphasparse_status_t hermm_z_dia_u_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)*B + beta*C
alphasparse_status_t hermm_z_dia_n_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)*B + beta*C
alphasparse_status_t hermm_z_dia_u_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*_plain(L+D)^T*B + beta*C
alphasparse_status_t hermm_z_dia_n_lo_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)^T*B + beta*C
alphasparse_status_t hermm_z_dia_u_lo_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)^T*B + beta*C
alphasparse_status_t hermm_z_dia_n_hi_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)^T*B + beta*C
alphasparse_status_t hermm_z_dia_u_hi_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+D)^T*B + beta*C
alphasparse_status_t hermm_z_dia_n_lo_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)^T*B + beta*C
alphasparse_status_t hermm_z_dia_u_lo_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)^T*B + beta*C
alphasparse_status_t hermm_z_dia_n_hi_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)^T*B + beta*C
alphasparse_status_t hermm_z_dia_u_hi_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*_plain(L+D)*B + beta*C
alphasparse_status_t trmm_z_dia_n_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)*B + beta*C
alphasparse_status_t trmm_z_dia_u_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op_plain(U+D)*B + beta*C
alphasparse_status_t trmm_z_dia_n_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*op_plain(U+I)*B + beta*C
alphasparse_status_t trmm_z_dia_u_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*_plain(L+D)*B + beta*C
alphasparse_status_t trmm_z_dia_n_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)*B + beta*C
alphasparse_status_t trmm_z_dia_u_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)*B + beta*C
alphasparse_status_t trmm_z_dia_n_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)*B + beta*C
alphasparse_status_t trmm_z_dia_u_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*_plain(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_lo_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_lo_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_hi_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_hi_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_lo_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_lo_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_hi_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_hi_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);


// alpha*_plain(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_lo_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_lo_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_hi_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_hi_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_lo_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(L+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_lo_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+D)^T*B + beta*C
alphasparse_status_t trmm_z_dia_n_hi_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*_plain(U+I)^T*B + beta*C
alphasparse_status_t trmm_z_dia_u_hi_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t diagmm_z_dia_n_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_z_dia_u_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t diagmm_z_dia_n_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_z_dia_u_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *mat, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Complex16 beta, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t spmmd_z_dia_row_plain(const spmat_dia_z_t *matA, const spmat_dia_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t spmmd_z_dia_col_plain(const spmat_dia_z_t *matA, const spmat_dia_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_dia_row_trans_plain(const spmat_dia_z_t *matA, const spmat_dia_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_dia_col_trans_plain(const spmat_dia_z_t *matA, const spmat_dia_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_dia_row_conj_plain(const spmat_dia_z_t *matA, const spmat_dia_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_z_dia_col_conj_plain(const spmat_dia_z_t *matA, const spmat_dia_z_t *matB, ALPHA_Complex16 *C, const ALPHA_INT ldc);

alphasparse_status_t spmm_z_dia_plain(const spmat_dia_z_t *A, const spmat_dia_z_t *B, spmat_dia_z_t **C);
alphasparse_status_t spmm_z_dia_trans_plain(const spmat_dia_z_t *A, const spmat_dia_z_t *B, spmat_dia_z_t **C);
alphasparse_status_t spmm_z_dia_conj_plain(const spmat_dia_z_t *A, const spmat_dia_z_t *B, spmat_dia_z_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t trsv_z_dia_n_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L)*x
alphasparse_status_t trsv_z_dia_u_lo_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_z_dia_n_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_z_dia_u_hi_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_dia_n_lo_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_dia_u_lo_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_dia_n_hi_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_dia_u_hi_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_dia_n_lo_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_z_dia_u_lo_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_dia_n_hi_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_z_dia_u_hi_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(D)*x
alphasparse_status_t diagsv_z_dia_n_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);
// alpha*x
alphasparse_status_t diagsv_z_dia_u_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, ALPHA_Complex16 *y);

// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_n_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_u_lo_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_n_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_u_hi_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_n_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_u_lo_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_n_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_u_hi_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_n_lo_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_u_lo_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_n_hi_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_u_hi_row_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_n_lo_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_u_lo_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_n_hi_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_u_hi_col_trans_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_n_lo_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_u_lo_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_n_hi_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_u_hi_row_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_n_lo_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_z_dia_u_lo_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_n_hi_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_z_dia_u_hi_col_conj_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t diagsm_z_dia_n_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_z_dia_u_row_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t diagsm_z_dia_n_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_z_dia_u_col_plain(const ALPHA_Complex16 alpha, const spmat_dia_z_t *A, const ALPHA_Complex16 *x, const ALPHA_INT columns, const ALPHA_INT ldx, ALPHA_Complex16 *y, const ALPHA_INT ldy);

alphasparse_status_t set_value_z_dia_plain (spmat_dia_z_t * A, const ALPHA_INT row, const ALPHA_INT col, const ALPHA_Complex16 value);
