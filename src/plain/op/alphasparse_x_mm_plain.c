/**
 * @brief implement for alphasparse_?_mm intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "alphasparse/spdef.h"

static alphasparse_status_t (*gemm_csr_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_CSR *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    gemm_csr_row_plain,
    gemm_csr_col_plain,
    gemm_csr_row_trans_plain,
    gemm_csr_col_trans_plain,
#ifdef COMPLEX
    gemm_csr_row_conj_plain,
    gemm_csr_col_conj_plain,
#endif
};

static alphasparse_status_t (*symm_csr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_CSR *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    symm_csr_n_lo_row_plain,
    symm_csr_u_lo_row_plain,
    symm_csr_n_hi_row_plain,
    symm_csr_u_hi_row_plain,
    symm_csr_n_lo_col_plain,
    symm_csr_u_lo_col_plain,
    symm_csr_n_hi_col_plain,
    symm_csr_u_hi_col_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_csr_n_lo_row_conj_plain,
    symm_csr_u_lo_row_conj_plain,
    symm_csr_n_hi_row_conj_plain,
    symm_csr_u_hi_row_conj_plain,
    symm_csr_n_lo_col_conj_plain,
    symm_csr_u_lo_col_conj_plain,
    symm_csr_n_hi_col_conj_plain,
    symm_csr_u_hi_col_conj_plain,
#endif
};

static alphasparse_status_t (*hermm_csr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                     const ALPHA_SPMAT_CSR *mat,
                                                     const ALPHA_Number *x,
                                                     const ALPHA_INT columns,
                                                     const ALPHA_INT ldx,
                                                     const ALPHA_Number beta,
                                                     ALPHA_Number *y,
                                                     const ALPHA_INT ldy) = {
#ifndef COMPLEX 
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding

    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermm_csr_n_lo_row_plain,
    hermm_csr_u_lo_row_plain,
    hermm_csr_n_hi_row_plain,
    hermm_csr_u_hi_row_plain,
    hermm_csr_n_lo_col_plain,
    hermm_csr_u_lo_col_plain,
    hermm_csr_n_hi_col_plain,
    hermm_csr_u_hi_col_plain,
    
    hermm_csr_n_lo_row_trans_plain,
    hermm_csr_u_lo_row_trans_plain,
    hermm_csr_n_hi_row_trans_plain,
    hermm_csr_u_hi_row_trans_plain,
    hermm_csr_n_lo_col_trans_plain,
    hermm_csr_u_lo_col_trans_plain,
    hermm_csr_n_hi_col_trans_plain,
    hermm_csr_u_hi_col_trans_plain,
#endif
};

static alphasparse_status_t (*trmm_csr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                   const ALPHA_SPMAT_CSR *mat,
                                                                   const ALPHA_Number *x,
                                                                   const ALPHA_INT columns,
                                                                   const ALPHA_INT ldx,
                                                                   const ALPHA_Number beta,
                                                                   ALPHA_Number *y,
                                                                   const ALPHA_INT ldy) = {
    trmm_csr_n_lo_row_plain,
    trmm_csr_u_lo_row_plain,
    trmm_csr_n_hi_row_plain,
    trmm_csr_u_hi_row_plain,
    trmm_csr_n_lo_col_plain,
    trmm_csr_u_lo_col_plain,
    trmm_csr_n_hi_col_plain,
    trmm_csr_u_hi_col_plain,

    trmm_csr_n_lo_row_trans_plain,
    trmm_csr_u_lo_row_trans_plain,
    trmm_csr_n_hi_row_trans_plain,
    trmm_csr_u_hi_row_trans_plain,
    trmm_csr_n_lo_col_trans_plain,
    trmm_csr_u_lo_col_trans_plain,
    trmm_csr_n_hi_col_trans_plain,
    trmm_csr_u_hi_col_trans_plain,
#ifdef COMPLEX
    trmm_csr_n_lo_row_conj_plain,
    trmm_csr_u_lo_row_conj_plain,
    trmm_csr_n_hi_row_conj_plain,
    trmm_csr_u_hi_row_conj_plain,
    trmm_csr_n_lo_col_conj_plain,
    trmm_csr_u_lo_col_conj_plain,
    trmm_csr_n_hi_col_conj_plain,
    trmm_csr_u_hi_col_conj_plain,
#endif
};

static alphasparse_status_t (*diagmm_csr_diag_layout_plain[])(const ALPHA_Number alpha,
                                                      const ALPHA_SPMAT_CSR *mat,
                                                      const ALPHA_Number *x,
                                                      const ALPHA_INT columns,
                                                      const ALPHA_INT ldx,
                                                      const ALPHA_Number beta,
                                                      ALPHA_Number *y,
                                                      const ALPHA_INT ldy) = {
    diagmm_csr_n_row_plain,
    diagmm_csr_u_row_plain,
    diagmm_csr_n_col_plain,
    diagmm_csr_u_col_plain,
};

static alphasparse_status_t (*gemm_coo_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_COO *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    gemm_coo_row_plain,
    gemm_coo_col_plain,
    gemm_coo_row_trans_plain,
    gemm_coo_col_trans_plain,
#ifdef COMPLEX
    gemm_coo_row_conj_plain,
    gemm_coo_col_conj_plain,
#endif 
};

static alphasparse_status_t (*symm_coo_diag_fill_layout_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_COO *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    symm_coo_n_lo_row_plain,
    symm_coo_u_lo_row_plain,
    symm_coo_n_hi_row_plain,
    symm_coo_u_hi_row_plain,
    symm_coo_n_lo_col_plain,
    symm_coo_u_lo_col_plain,
    symm_coo_n_hi_col_plain,
    symm_coo_u_hi_col_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_coo_n_lo_row_conj_plain,
    symm_coo_u_lo_row_conj_plain,
    symm_coo_n_hi_row_conj_plain,
    symm_coo_u_hi_row_conj_plain,
    symm_coo_n_lo_col_conj_plain,
    symm_coo_u_lo_col_conj_plain,
    symm_coo_n_hi_col_conj_plain,
    symm_coo_u_hi_col_conj_plain,
#endif
};

static alphasparse_status_t (*hermm_coo_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                     const ALPHA_SPMAT_COO *mat,
                                                     const ALPHA_Number *x,
                                                     const ALPHA_INT columns,
                                                     const ALPHA_INT ldx,
                                                     const ALPHA_Number beta,
                                                     ALPHA_Number *y,
                                                     const ALPHA_INT ldy) = {
#ifndef COMPLEX 
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding

    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermm_coo_n_lo_row_plain,
    hermm_coo_u_lo_row_plain,
    hermm_coo_n_hi_row_plain,
    hermm_coo_u_hi_row_plain,
    hermm_coo_n_lo_col_plain,
    hermm_coo_u_lo_col_plain,
    hermm_coo_n_hi_col_plain,
    hermm_coo_u_hi_col_plain,
    
    hermm_coo_n_lo_row_trans_plain,
    hermm_coo_u_lo_row_trans_plain,
    hermm_coo_n_hi_row_trans_plain,
    hermm_coo_u_hi_row_trans_plain,
    hermm_coo_n_lo_col_trans_plain,
    hermm_coo_u_lo_col_trans_plain,
    hermm_coo_n_hi_col_trans_plain,
    hermm_coo_u_hi_col_trans_plain,
#endif
};

static alphasparse_status_t (*trmm_coo_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                   const ALPHA_SPMAT_COO *mat,
                                                                   const ALPHA_Number *x,
                                                                   const ALPHA_INT columns,
                                                                   const ALPHA_INT ldx,
                                                                   const ALPHA_Number beta,
                                                                   ALPHA_Number *y,
                                                                   const ALPHA_INT ldy) = {
    trmm_coo_n_lo_row_plain,
    trmm_coo_u_lo_row_plain,
    trmm_coo_n_hi_row_plain,
    trmm_coo_u_hi_row_plain,
    trmm_coo_n_lo_col_plain,
    trmm_coo_u_lo_col_plain,
    trmm_coo_n_hi_col_plain,
    trmm_coo_u_hi_col_plain,

    trmm_coo_n_lo_row_trans_plain,
    trmm_coo_u_lo_row_trans_plain,
    trmm_coo_n_hi_row_trans_plain,
    trmm_coo_u_hi_row_trans_plain,
    trmm_coo_n_lo_col_trans_plain,
    trmm_coo_u_lo_col_trans_plain,
    trmm_coo_n_hi_col_trans_plain,
    trmm_coo_u_hi_col_trans_plain,

#ifdef COMPLEX
    trmm_coo_n_lo_row_conj_plain,
    trmm_coo_u_lo_row_conj_plain,
    trmm_coo_n_hi_row_conj_plain,
    trmm_coo_u_hi_row_conj_plain,
    trmm_coo_n_lo_col_conj_plain,
    trmm_coo_u_lo_col_conj_plain,
    trmm_coo_n_hi_col_conj_plain,
    trmm_coo_u_hi_col_conj_plain,
#endif
};

static alphasparse_status_t (*diagmm_coo_diag_layout_plain[])(const ALPHA_Number alpha,
                                                      const ALPHA_SPMAT_COO *mat,
                                                      const ALPHA_Number *x,
                                                      const ALPHA_INT columns,
                                                      const ALPHA_INT ldx,
                                                      const ALPHA_Number beta,
                                                      ALPHA_Number *y,
                                                      const ALPHA_INT ldy) = {
    diagmm_coo_n_row_plain,
    diagmm_coo_u_row_plain,
    diagmm_coo_n_col_plain,
    diagmm_coo_u_col_plain,
};

static alphasparse_status_t (*gemm_csc_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_CSC *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    gemm_csc_row_plain,
    gemm_csc_col_plain,
    gemm_csc_row_trans_plain,
    gemm_csc_col_trans_plain,
#ifdef COMPLEX
    gemm_csc_row_conj_plain,
    gemm_csc_col_conj_plain,
#endif
};

static alphasparse_status_t (*symm_csc_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_CSC *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    symm_csc_n_lo_row_plain,
    symm_csc_u_lo_row_plain,
    symm_csc_n_hi_row_plain,
    symm_csc_u_hi_row_plain,
    symm_csc_n_lo_col_plain,
    symm_csc_u_lo_col_plain,
    symm_csc_n_hi_col_plain,
    symm_csc_u_hi_col_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_csc_n_lo_row_conj_plain,
    symm_csc_u_lo_row_conj_plain,
    symm_csc_n_hi_row_conj_plain,
    symm_csc_u_hi_row_conj_plain,
    symm_csc_n_lo_col_conj_plain,
    symm_csc_u_lo_col_conj_plain,
    symm_csc_n_hi_col_conj_plain,
    symm_csc_u_hi_col_conj_plain,
#endif
};

static alphasparse_status_t (*hermm_csc_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                     const ALPHA_SPMAT_CSC *mat,
                                                     const ALPHA_Number *x,
                                                     const ALPHA_INT columns,
                                                     const ALPHA_INT ldx,
                                                     const ALPHA_Number beta,
                                                     ALPHA_Number *y,
                                                     const ALPHA_INT ldy) = {
#ifndef COMPLEX 
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding

    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermm_csc_n_lo_row_plain,
    hermm_csc_u_lo_row_plain,
    hermm_csc_n_hi_row_plain,
    hermm_csc_u_hi_row_plain,
    hermm_csc_n_lo_col_plain,
    hermm_csc_u_lo_col_plain,
    hermm_csc_n_hi_col_plain,
    hermm_csc_u_hi_col_plain,
    
    hermm_csc_n_lo_row_trans_plain,
    hermm_csc_u_lo_row_trans_plain,
    hermm_csc_n_hi_row_trans_plain,
    hermm_csc_u_hi_row_trans_plain,
    hermm_csc_n_lo_col_trans_plain,
    hermm_csc_u_lo_col_trans_plain,
    hermm_csc_n_hi_col_trans_plain,
    hermm_csc_u_hi_col_trans_plain,
#endif
};

static alphasparse_status_t (*trmm_csc_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                   const ALPHA_SPMAT_CSC *mat,
                                                                   const ALPHA_Number *x,
                                                                   const ALPHA_INT columns,
                                                                   const ALPHA_INT ldx,
                                                                   const ALPHA_Number beta,
                                                                   ALPHA_Number *y,
                                                                   const ALPHA_INT ldy) = {
    trmm_csc_n_lo_row_plain,
    trmm_csc_u_lo_row_plain,
    trmm_csc_n_hi_row_plain,
    trmm_csc_u_hi_row_plain,
    trmm_csc_n_lo_col_plain,
    trmm_csc_u_lo_col_plain,
    trmm_csc_n_hi_col_plain,
    trmm_csc_u_hi_col_plain,

    trmm_csc_n_lo_row_trans_plain,
    trmm_csc_u_lo_row_trans_plain,
    trmm_csc_n_hi_row_trans_plain,
    trmm_csc_u_hi_row_trans_plain,
    trmm_csc_n_lo_col_trans_plain,
    trmm_csc_u_lo_col_trans_plain,
    trmm_csc_n_hi_col_trans_plain,
    trmm_csc_u_hi_col_trans_plain,
#ifdef COMPLEX
    trmm_csc_n_lo_row_conj_plain,
    trmm_csc_u_lo_row_conj_plain,
    trmm_csc_n_hi_row_conj_plain,
    trmm_csc_u_hi_row_conj_plain,
    trmm_csc_n_lo_col_conj_plain,
    trmm_csc_u_lo_col_conj_plain,
    trmm_csc_n_hi_col_conj_plain,
    trmm_csc_u_hi_col_conj_plain,
#endif
};

static alphasparse_status_t (*diagmm_csc_diag_layout_plain[])(const ALPHA_Number alpha,
                                                      const ALPHA_SPMAT_CSC *mat,
                                                      const ALPHA_Number *x,
                                                      const ALPHA_INT columns,
                                                      const ALPHA_INT ldx,
                                                      const ALPHA_Number beta,
                                                      ALPHA_Number *y,
                                                      const ALPHA_INT ldy) = {
    diagmm_csc_n_row_plain,
    diagmm_csc_u_row_plain,
    diagmm_csc_n_col_plain,
    diagmm_csc_u_col_plain,
};

static alphasparse_status_t (*gemm_bsr_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_BSR *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    gemm_bsr_row_plain,
    gemm_bsr_col_plain,
    gemm_bsr_row_trans_plain,
    gemm_bsr_col_trans_plain,
#ifdef COMPLEX
    gemm_bsr_row_conj_plain,
    gemm_bsr_col_conj_plain,
#endif
};
static alphasparse_status_t (*symm_bsr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_BSR *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    symm_bsr_n_lo_row_plain,
    symm_bsr_u_lo_row_plain,
    symm_bsr_n_hi_row_plain,
    symm_bsr_u_hi_row_plain,
    symm_bsr_n_lo_col_plain,
    symm_bsr_u_lo_col_plain,
    symm_bsr_n_hi_col_plain,
    symm_bsr_u_hi_col_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_bsr_n_lo_row_conj_plain,
    symm_bsr_u_lo_row_conj_plain,
    symm_bsr_n_hi_row_conj_plain,
    symm_bsr_u_hi_row_conj_plain,
    symm_bsr_n_lo_col_conj_plain,
    symm_bsr_u_lo_col_conj_plain,
    symm_bsr_n_hi_col_conj_plain,
    symm_bsr_u_hi_col_conj_plain,
#endif
};

static alphasparse_status_t (*hermm_bsr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                     const ALPHA_SPMAT_BSR *mat,
                                                     const ALPHA_Number *x,
                                                     const ALPHA_INT columns,
                                                     const ALPHA_INT ldx,
                                                     const ALPHA_Number beta,
                                                     ALPHA_Number *y,
                                                     const ALPHA_INT ldy) = {
#ifndef COMPLEX 
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding

    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermm_bsr_n_lo_row_plain,
    hermm_bsr_u_lo_row_plain,
    hermm_bsr_n_hi_row_plain,
    hermm_bsr_u_hi_row_plain,
    hermm_bsr_n_lo_col_plain,
    hermm_bsr_u_lo_col_plain,
    hermm_bsr_n_hi_col_plain,
    hermm_bsr_u_hi_col_plain,
    
    hermm_bsr_n_lo_row_trans_plain,
    hermm_bsr_u_lo_row_trans_plain,
    hermm_bsr_n_hi_row_trans_plain,
    hermm_bsr_u_hi_row_trans_plain,
    hermm_bsr_n_lo_col_trans_plain,
    hermm_bsr_u_lo_col_trans_plain,
    hermm_bsr_n_hi_col_trans_plain,
    hermm_bsr_u_hi_col_trans_plain,
#endif
};

static alphasparse_status_t (*trmm_bsr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                   const ALPHA_SPMAT_BSR *mat,
                                                                   const ALPHA_Number *x,
                                                                   const ALPHA_INT columns,
                                                                   const ALPHA_INT ldx,
                                                                   const ALPHA_Number beta,
                                                                   ALPHA_Number *y,
                                                                   const ALPHA_INT ldy) = {
    trmm_bsr_n_lo_row_plain,
    trmm_bsr_u_lo_row_plain,
    trmm_bsr_n_hi_row_plain,
    trmm_bsr_u_hi_row_plain,
    trmm_bsr_n_lo_col_plain,
    trmm_bsr_u_lo_col_plain,
    trmm_bsr_n_hi_col_plain,
    trmm_bsr_u_hi_col_plain,
    trmm_bsr_n_lo_row_trans_plain,
    trmm_bsr_u_lo_row_trans_plain,
    trmm_bsr_n_hi_row_trans_plain,
    trmm_bsr_u_hi_row_trans_plain,
    trmm_bsr_n_lo_col_trans_plain,
    trmm_bsr_u_lo_col_trans_plain,
    trmm_bsr_n_hi_col_trans_plain,
    trmm_bsr_u_hi_col_trans_plain,
#ifdef COMPLEX
    trmm_bsr_n_lo_row_conj_plain,
    trmm_bsr_u_lo_row_conj_plain,
    trmm_bsr_n_hi_row_conj_plain,
    trmm_bsr_u_hi_row_conj_plain,
    trmm_bsr_n_lo_col_conj_plain,
    trmm_bsr_u_lo_col_conj_plain,
    trmm_bsr_n_hi_col_conj_plain,
    trmm_bsr_u_hi_col_conj_plain,
#endif
};
static alphasparse_status_t (*diagmm_bsr_diag_layout_plain[])(const ALPHA_Number alpha,
                                                      const ALPHA_SPMAT_BSR *mat,
                                                      const ALPHA_Number *x,
                                                      const ALPHA_INT columns,
                                                      const ALPHA_INT ldx,
                                                      const ALPHA_Number beta,
                                                      ALPHA_Number *y,
                                                      const ALPHA_INT ldy) = {
    diagmm_bsr_n_row_plain,
    diagmm_bsr_u_row_plain,
    diagmm_bsr_n_col_plain,
    diagmm_bsr_u_col_plain,
};

static alphasparse_status_t (*gemm_sky_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_SKY *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    gemm_sky_row_plain,
    gemm_sky_col_plain,
    gemm_sky_row_trans_plain,
    gemm_sky_col_trans_plain,
#ifdef COMPLEX
    NULL,
    NULL,
#endif
};
static alphasparse_status_t (*symm_sky_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_SKY *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    symm_sky_n_lo_row_plain,
    symm_sky_u_lo_row_plain,
    symm_sky_n_hi_row_plain,
    symm_sky_u_hi_row_plain,
    symm_sky_n_lo_col_plain,
    symm_sky_u_lo_col_plain,
    symm_sky_n_hi_col_plain,
    symm_sky_u_hi_col_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_sky_n_lo_row_conj_plain,
    symm_sky_u_lo_row_conj_plain,
    symm_sky_n_hi_row_conj_plain,
    symm_sky_u_hi_row_conj_plain,
    symm_sky_n_lo_col_conj_plain,
    symm_sky_u_lo_col_conj_plain,
    symm_sky_n_hi_col_conj_plain,
    symm_sky_u_hi_col_conj_plain,
#endif
};
static alphasparse_status_t (*hermm_sky_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                     const ALPHA_SPMAT_SKY *mat,
                                                     const ALPHA_Number *x,
                                                     const ALPHA_INT columns,
                                                     const ALPHA_INT ldx,
                                                     const ALPHA_Number beta,
                                                     ALPHA_Number *y,
                                                     const ALPHA_INT ldy) = {
#ifndef COMPLEX 
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding

    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermm_sky_n_lo_row_plain,
    hermm_sky_u_lo_row_plain,
    hermm_sky_n_hi_row_plain,
    hermm_sky_u_hi_row_plain,
    hermm_sky_n_lo_col_plain,
    hermm_sky_u_lo_col_plain,
    hermm_sky_n_hi_col_plain,
    hermm_sky_u_hi_col_plain,
    
    hermm_sky_n_lo_row_trans_plain,
    hermm_sky_u_lo_row_trans_plain,
    hermm_sky_n_hi_row_trans_plain,
    hermm_sky_u_hi_row_trans_plain,
    hermm_sky_n_lo_col_trans_plain,
    hermm_sky_u_lo_col_trans_plain,
    hermm_sky_n_hi_col_trans_plain,
    hermm_sky_u_hi_col_trans_plain,
#endif
};
static alphasparse_status_t (*trmm_sky_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                   const ALPHA_SPMAT_SKY *mat,
                                                                   const ALPHA_Number *x,
                                                                   const ALPHA_INT columns,
                                                                   const ALPHA_INT ldx,
                                                                   const ALPHA_Number beta,
                                                                   ALPHA_Number *y,
                                                                   const ALPHA_INT ldy) = {
    trmm_sky_n_lo_row_plain,
    trmm_sky_u_lo_row_plain,
    trmm_sky_n_hi_row_plain,
    trmm_sky_u_hi_row_plain,
    trmm_sky_n_lo_col_plain,
    trmm_sky_u_lo_col_plain,
    trmm_sky_n_hi_col_plain,
    trmm_sky_u_hi_col_plain,
    trmm_sky_n_lo_row_trans_plain,
    trmm_sky_u_lo_row_trans_plain,
    trmm_sky_n_hi_row_trans_plain,
    trmm_sky_u_hi_row_trans_plain,
    trmm_sky_n_lo_col_trans_plain,
    trmm_sky_u_lo_col_trans_plain,
    trmm_sky_n_hi_col_trans_plain,
    trmm_sky_u_hi_col_trans_plain,
#ifdef COMPLEX
    trmm_sky_n_lo_row_conj_plain,
    trmm_sky_u_lo_row_conj_plain,
    trmm_sky_n_hi_row_conj_plain,
    trmm_sky_u_hi_row_conj_plain,
    trmm_sky_n_lo_col_conj_plain,
    trmm_sky_u_lo_col_conj_plain,
    trmm_sky_n_hi_col_conj_plain,
    trmm_sky_u_hi_col_conj_plain,
#endif
};
static alphasparse_status_t (*diagmm_sky_diag_layout_plain[])(const ALPHA_Number alpha,
                                                      const ALPHA_SPMAT_SKY *mat,
                                                      const ALPHA_Number *x,
                                                      const ALPHA_INT columns,
                                                      const ALPHA_INT ldx,
                                                      const ALPHA_Number beta,
                                                      ALPHA_Number *y,
                                                      const ALPHA_INT ldy) = {
    diagmm_sky_n_row_plain,
    diagmm_sky_u_row_plain,
    diagmm_sky_n_col_plain,
    diagmm_sky_u_col_plain,
};

static alphasparse_status_t (*gemm_dia_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_DIA *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    gemm_dia_row_plain,
    gemm_dia_col_plain,
    gemm_dia_row_trans_plain,
    gemm_dia_col_trans_plain,
#ifdef COMPLEX
    gemm_dia_row_conj_plain,
    gemm_dia_col_conj_plain,
#endif
};
static alphasparse_status_t (*symm_dia_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                         const ALPHA_SPMAT_DIA *mat,
                                                         const ALPHA_Number *x,
                                                         const ALPHA_INT columns,
                                                         const ALPHA_INT ldx,
                                                         const ALPHA_Number beta,
                                                         ALPHA_Number *y,
                                                         const ALPHA_INT ldy) = {
    symm_dia_n_lo_row_plain,
    symm_dia_u_lo_row_plain,
    symm_dia_n_hi_row_plain,
    symm_dia_u_hi_row_plain,
    symm_dia_n_lo_col_plain,
    symm_dia_u_lo_col_plain,
    symm_dia_n_hi_col_plain,
    symm_dia_u_hi_col_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_dia_n_lo_row_conj_plain,
    symm_dia_u_lo_row_conj_plain,
    symm_dia_n_hi_row_conj_plain,
    symm_dia_u_hi_row_conj_plain,
    symm_dia_n_lo_col_conj_plain,
    symm_dia_u_lo_col_conj_plain,
    symm_dia_n_hi_col_conj_plain,
    symm_dia_u_hi_col_conj_plain,
#endif
};
static alphasparse_status_t (*hermm_dia_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                     const ALPHA_SPMAT_DIA *mat,
                                                     const ALPHA_Number *x,
                                                     const ALPHA_INT columns,
                                                     const ALPHA_INT ldx,
                                                     const ALPHA_Number beta,
                                                     ALPHA_Number *y,
                                                     const ALPHA_INT ldy) = {
#ifndef COMPLEX 
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding

    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermm_dia_n_lo_row_plain,
    hermm_dia_u_lo_row_plain,
    hermm_dia_n_hi_row_plain,
    hermm_dia_u_hi_row_plain,
    hermm_dia_n_lo_col_plain,
    hermm_dia_u_lo_col_plain,
    hermm_dia_n_hi_col_plain,
    hermm_dia_u_hi_col_plain,
    
    hermm_dia_n_lo_row_trans_plain,
    hermm_dia_u_lo_row_trans_plain,
    hermm_dia_n_hi_row_trans_plain,
    hermm_dia_u_hi_row_trans_plain,
    hermm_dia_n_lo_col_trans_plain,
    hermm_dia_u_lo_col_trans_plain,
    hermm_dia_n_hi_col_trans_plain,
    hermm_dia_u_hi_col_trans_plain,
#endif
};
static alphasparse_status_t (*trmm_dia_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                   const ALPHA_SPMAT_DIA *mat,
                                                                   const ALPHA_Number *x,
                                                                   const ALPHA_INT columns,
                                                                   const ALPHA_INT ldx,
                                                                   const ALPHA_Number beta,
                                                                   ALPHA_Number *y,
                                                                   const ALPHA_INT ldy) = {
    trmm_dia_n_lo_row_plain,
    trmm_dia_u_lo_row_plain,
    trmm_dia_n_hi_row_plain,
    trmm_dia_u_hi_row_plain,
    trmm_dia_n_lo_col_plain,
    trmm_dia_u_lo_col_plain,
    trmm_dia_n_hi_col_plain,
    trmm_dia_u_hi_col_plain,
    trmm_dia_n_lo_row_trans_plain,
    trmm_dia_u_lo_row_trans_plain,
    trmm_dia_n_hi_row_trans_plain,
    trmm_dia_u_hi_row_trans_plain,
    trmm_dia_n_lo_col_trans_plain,
    trmm_dia_u_lo_col_trans_plain,
    trmm_dia_n_hi_col_trans_plain,
    trmm_dia_u_hi_col_trans_plain,
#ifdef COMPLEX
    trmm_dia_n_lo_row_conj_plain,
    trmm_dia_u_lo_row_conj_plain,
    trmm_dia_n_hi_row_conj_plain,
    trmm_dia_u_hi_row_conj_plain,
    trmm_dia_n_lo_col_conj_plain,
    trmm_dia_u_lo_col_conj_plain,
    trmm_dia_n_hi_col_conj_plain,
    trmm_dia_u_hi_col_conj_plain,
#endif
};
static alphasparse_status_t (*diagmm_dia_diag_layout_plain[])(const ALPHA_Number alpha,
                                                      const ALPHA_SPMAT_DIA *mat,
                                                      const ALPHA_Number *x,
                                                      const ALPHA_INT columns,
                                                      const ALPHA_INT ldx,
                                                      const ALPHA_Number beta,
                                                      ALPHA_Number *y,
                                                      const ALPHA_INT ldy) = {
    diagmm_dia_n_row_plain,
    diagmm_dia_u_row_plain,
    diagmm_dia_n_col_plain,
    diagmm_dia_u_col_plain,
};

alphasparse_status_t ONAME(const alphasparse_operation_t operation,
                          const ALPHA_Number alpha,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                          const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                          const ALPHA_Number *x,
                          const ALPHA_INT columns,
                          const ALPHA_INT ldx,
                          const ALPHA_Number beta,
                          ALPHA_Number *y,
                          const ALPHA_INT ldy)
{
    check_null_return(A, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
    if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif

    if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC || descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        // check if it is a square matrix 
        check_return(!check_equal_row_col(A),ALPHA_SPARSE_STATUS_INVALID_VALUE);

    if (A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_csr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_csr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_csr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_csr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_COO)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_coo_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_coo_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_coo_diag_fill_layout_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_coo_diag_fill_layout_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_coo_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_coo_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_coo_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_coo_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_coo_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_coo_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_csc_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_csc_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_csc_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_csc_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_bsr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_bsr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_bsr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_bsr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_SKY)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_sky_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_sky_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_DIA)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_dia_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_dia_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_dia_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_dia_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else
    {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
}
