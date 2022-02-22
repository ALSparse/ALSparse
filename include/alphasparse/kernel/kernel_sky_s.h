#pragma once

#include "../spmat.h"

alphasparse_status_t add_s_sky(const spmat_sky_s_t *A, const float alpha, const spmat_sky_s_t *B, spmat_sky_s_t **C);
alphasparse_status_t add_s_sky_trans(const spmat_sky_s_t *A, const float alpha, const spmat_sky_s_t *B, spmat_sky_s_t **C);

// --------------------------------------------------------------------------------------------------------------------------------

// mv
// alpha*A*x + beta*y
alphasparse_status_t gemv_s_sky(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*A^T*x + beta*y
alphasparse_status_t gemv_s_sky_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);

// alpha*(L+D+L')*x + beta*y
alphasparse_status_t symv_s_sky_n_lo(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(L+I+L')*x + beta*y
alphasparse_status_t symv_s_sky_u_lo(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(U'+D+U)*x + beta*y
alphasparse_status_t symv_s_sky_n_hi(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(U'+I+U)*x + beta*y
alphasparse_status_t symv_s_sky_u_hi(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);

// alpha*(L+D)*x + beta*y
alphasparse_status_t trmv_s_sky_n_lo(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(L+I)*x + beta*y
alphasparse_status_t trmv_s_sky_u_lo(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(U+D)*x + beta*y
alphasparse_status_t trmv_s_sky_n_hi(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(U+I)*x + beta*y
alphasparse_status_t trmv_s_sky_u_hi(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);

// alpha*(L+D)^T*x + beta*y
alphasparse_status_t trmv_s_sky_n_lo_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(L+I)^T*x + beta*y
alphasparse_status_t trmv_s_sky_u_lo_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(U+D)^T*x + beta*y
alphasparse_status_t trmv_s_sky_n_hi_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*(U+I)^T*x + beta*y
alphasparse_status_t trmv_s_sky_u_hi_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);

// alpha*D*x + beta*y
alphasparse_status_t diagmv_s_sky_n(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);
// alpha*x + beta*y
alphasparse_status_t diagmv_s_sky_u(const float alpha, const spmat_sky_s_t *A, const float *x, const float beta, float *y);

// --------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------------

// alpha*A*B + beta*C
alphasparse_status_t gemm_s_sky_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_s_sky_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*A^T*B + beta*C
alphasparse_status_t gemm_s_sky_row_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
alphasparse_status_t gemm_s_sky_col_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_s_sky_n_lo_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_s_sky_u_lo_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_s_sky_n_hi_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_s_sky_u_hi_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// alpha*（L+D+L')^T*B + beta*C
alphasparse_status_t symm_s_sky_n_lo_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(L+I+L')*B + beta*C
alphasparse_status_t symm_s_sky_u_lo_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U'+D+U)*B + beta*C
alphasparse_status_t symm_s_sky_n_hi_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U'+I+U)*B + beta*C
alphasparse_status_t symm_s_sky_u_hi_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_s_sky_n_lo_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_s_sky_u_lo_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*op(U+D)*B + beta*C
alphasparse_status_t trmm_s_sky_n_hi_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*op(U+I)*B + beta*C
alphasparse_status_t trmm_s_sky_u_hi_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// alpha*(L+D)*B + beta*C
alphasparse_status_t trmm_s_sky_n_lo_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(L+I)*B + beta*C
alphasparse_status_t trmm_s_sky_u_lo_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U+D)*B + beta*C
alphasparse_status_t trmm_s_sky_n_hi_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U+I)*B + beta*C
alphasparse_status_t trmm_s_sky_u_hi_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_s_sky_n_lo_row_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_s_sky_u_lo_row_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_s_sky_n_hi_row_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_s_sky_u_hi_row_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(L+D)^T*B + beta*C
alphasparse_status_t trmm_s_sky_n_lo_col_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(L+I)^T*B + beta*C
alphasparse_status_t trmm_s_sky_u_lo_col_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U+D)^T*B + beta*C
alphasparse_status_t trmm_s_sky_n_hi_col_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*(U+I)^T*B + beta*C
alphasparse_status_t trmm_s_sky_u_hi_col_trans(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// alpha*D*B + beta*C
alphasparse_status_t diagmm_s_sky_n_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_s_sky_u_row(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*D*B + beta*C
alphasparse_status_t diagmm_s_sky_n_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);
// alpha*B + beta*C
alphasparse_status_t diagmm_s_sky_u_col(const float alpha, const spmat_sky_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy);

// ---------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------

// A*B
alphasparse_status_t spmmd_s_sky_row(const spmat_sky_s_t *matA, const spmat_sky_s_t *matB, float *C, const ALPHA_INT ldc);
// A*B
alphasparse_status_t spmmd_s_sky_col(const spmat_sky_s_t *matA, const spmat_sky_s_t *matB, float *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_s_sky_row_trans(const spmat_sky_s_t *matA, const spmat_sky_s_t *matB, float *C, const ALPHA_INT ldc);
// A^T*B
alphasparse_status_t spmmd_s_sky_col_trans(const spmat_sky_s_t *matA, const spmat_sky_s_t *matB, float *C, const ALPHA_INT ldc);

alphasparse_status_t spmm_s_sky(const spmat_sky_s_t *A, const spmat_sky_s_t *B, spmat_sky_s_t **C);
alphasparse_status_t spmm_s_sky_trans(const spmat_sky_s_t *A, const spmat_sky_s_t *B, spmat_sky_s_t **C);

// -----------------------------------------------------------------------------------------------------

// alpha*inv(L)*x
alphasparse_status_t trsv_s_sky_n_lo(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*inv(L)*x
alphasparse_status_t trsv_s_sky_u_lo(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_s_sky_n_hi(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*inv(U)*x
alphasparse_status_t trsv_s_sky_u_hi(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_s_sky_n_lo_trans(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*inv(L^T)*x
alphasparse_status_t trsv_s_sky_u_lo_trans(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_s_sky_n_hi_trans(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*inv(U^T)*x
alphasparse_status_t trsv_s_sky_u_hi_trans(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);

// alpha*inv(D)*x
alphasparse_status_t diagsv_s_sky_n(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);
// alpha*x
alphasparse_status_t diagsv_s_sky_u(const float alpha, const spmat_sky_s_t *A, const float *x, float *y);

// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_n_lo_row(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_u_lo_row(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_n_hi_row(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_u_hi_row(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_n_lo_col(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_u_lo_col(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_n_hi_col(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_u_hi_col(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);

// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_n_lo_row_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_u_lo_row_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_n_hi_row_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_u_hi_row_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_n_lo_col_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(L)*B
alphasparse_status_t trsm_s_sky_u_lo_col_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_n_hi_col_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(U)*B
alphasparse_status_t trsm_s_sky_u_hi_col_trans(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);

// alpha*inv(D)*x
alphasparse_status_t diagsm_s_sky_n_row(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_s_sky_u_row(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*inv(D)*x
alphasparse_status_t diagsm_s_sky_n_col(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);
// alpha*x
alphasparse_status_t diagsm_s_sky_u_col(const float alpha, const spmat_sky_s_t *A, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, float *y, const ALPHA_INT ldy);

alphasparse_status_t set_value_s_sky (spmat_sky_s_t * A, const ALPHA_INT row, const ALPHA_INT col, const float value);