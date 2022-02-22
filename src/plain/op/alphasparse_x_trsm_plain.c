/**
 * @brief implement for alphasparse_?_trsm intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"

static alphasparse_status_t (*trsm_csr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                     const ALPHA_SPMAT_CSR *A,
                                                                     const ALPHA_Number *x,
                                                                     const ALPHA_INT columns,
                                                                     const ALPHA_INT ldx,
                                                                     ALPHA_Number *y,
                                                                     const ALPHA_INT ldy) = {
    trsm_csr_n_lo_row_plain,
    trsm_csr_u_lo_row_plain,
    trsm_csr_n_hi_row_plain,
    trsm_csr_u_hi_row_plain,
    trsm_csr_n_lo_col_plain,
    trsm_csr_u_lo_col_plain,
    trsm_csr_n_hi_col_plain,
    trsm_csr_u_hi_col_plain,
    trsm_csr_n_lo_row_trans_plain,
    trsm_csr_u_lo_row_trans_plain,
    trsm_csr_n_hi_row_trans_plain,
    trsm_csr_u_hi_row_trans_plain,
    trsm_csr_n_lo_col_trans_plain,
    trsm_csr_u_lo_col_trans_plain,
    trsm_csr_n_hi_col_trans_plain,
    trsm_csr_u_hi_col_trans_plain,
#ifdef COMPLEX
    trsm_csr_n_lo_row_conj_plain, 
    trsm_csr_u_lo_row_conj_plain, 
    trsm_csr_n_hi_row_conj_plain, 
    trsm_csr_u_hi_row_conj_plain, 
    trsm_csr_n_lo_col_conj_plain, 
    trsm_csr_u_lo_col_conj_plain, 
    trsm_csr_n_hi_col_conj_plain, 
    trsm_csr_u_hi_col_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsm_csr_diag_layout_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_CSR *A,
                                                        const ALPHA_Number *x,
                                                        const ALPHA_INT columns,
                                                        const ALPHA_INT ldx,
                                                        ALPHA_Number *y,
                                                        const ALPHA_INT ldy) = {
    diagsm_csr_n_row_plain,
    diagsm_csr_u_row_plain,
    diagsm_csr_n_col_plain,
    diagsm_csr_u_col_plain,
};


static alphasparse_status_t (*trsm_csc_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                     const ALPHA_SPMAT_CSC *A,
                                                                     const ALPHA_Number *x,
                                                                     const ALPHA_INT columns,
                                                                     const ALPHA_INT ldx,
                                                                     ALPHA_Number *y,
                                                                     const ALPHA_INT ldy) = {
    trsm_csc_n_lo_row_plain,
    trsm_csc_u_lo_row_plain,
    trsm_csc_n_hi_row_plain,
    trsm_csc_u_hi_row_plain,
    trsm_csc_n_lo_col_plain,
    trsm_csc_u_lo_col_plain,
    trsm_csc_n_hi_col_plain,
    trsm_csc_u_hi_col_plain,
    trsm_csc_n_lo_row_trans_plain,
    trsm_csc_u_lo_row_trans_plain,
    trsm_csc_n_hi_row_trans_plain,
    trsm_csc_u_hi_row_trans_plain,
    trsm_csc_n_lo_col_trans_plain,
    trsm_csc_u_lo_col_trans_plain,
    trsm_csc_n_hi_col_trans_plain,
    trsm_csc_u_hi_col_trans_plain,
#ifdef COMPLEX
    trsm_csc_n_lo_row_conj_plain, 
    trsm_csc_u_lo_row_conj_plain, 
    trsm_csc_n_hi_row_conj_plain, 
    trsm_csc_u_hi_row_conj_plain, 
    trsm_csc_n_lo_col_conj_plain, 
    trsm_csc_u_lo_col_conj_plain, 
    trsm_csc_n_hi_col_conj_plain, 
    trsm_csc_u_hi_col_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsm_csc_diag_layout_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_CSC *A,
                                                        const ALPHA_Number *x,
                                                        const ALPHA_INT columns,
                                                        const ALPHA_INT ldx,
                                                        ALPHA_Number *y,
                                                        const ALPHA_INT ldy) = {
    diagsm_csc_n_row_plain,
    diagsm_csc_u_row_plain,
    diagsm_csc_n_col_plain,
    diagsm_csc_u_col_plain,
};

static alphasparse_status_t (*trsm_coo_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                     const ALPHA_SPMAT_COO *A,
                                                                     const ALPHA_Number *x,
                                                                     const ALPHA_INT columns,
                                                                     const ALPHA_INT ldx,
                                                                     ALPHA_Number *y,
                                                                     const ALPHA_INT ldy) = {
    trsm_coo_n_lo_row_plain,
    trsm_coo_u_lo_row_plain,
    trsm_coo_n_hi_row_plain,
    trsm_coo_u_hi_row_plain,
    trsm_coo_n_lo_col_plain,
    trsm_coo_u_lo_col_plain,
    trsm_coo_n_hi_col_plain,
    trsm_coo_u_hi_col_plain,
    trsm_coo_n_lo_row_trans_plain,
    trsm_coo_u_lo_row_trans_plain,
    trsm_coo_n_hi_row_trans_plain,
    trsm_coo_u_hi_row_trans_plain,
    trsm_coo_n_lo_col_trans_plain,
    trsm_coo_u_lo_col_trans_plain,
    trsm_coo_n_hi_col_trans_plain,
    trsm_coo_u_hi_col_trans_plain,
#ifdef COMPLEX
    trsm_coo_n_lo_row_conj_plain, 
    trsm_coo_u_lo_row_conj_plain, 
    trsm_coo_n_hi_row_conj_plain, 
    trsm_coo_u_hi_row_conj_plain, 
    trsm_coo_n_lo_col_conj_plain, 
    trsm_coo_u_lo_col_conj_plain, 
    trsm_coo_n_hi_col_conj_plain, 
    trsm_coo_u_hi_col_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsm_coo_diag_layout_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_COO *A,
                                                        const ALPHA_Number *x,
                                                        const ALPHA_INT columns,
                                                        const ALPHA_INT ldx,
                                                        ALPHA_Number *y,
                                                        const ALPHA_INT ldy) = {
    diagsm_coo_n_row_plain,
    diagsm_coo_u_row_plain,
    diagsm_coo_n_col_plain,
    diagsm_coo_u_col_plain,
};

static alphasparse_status_t (*trsm_bsr_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                     const ALPHA_SPMAT_BSR *A,
                                                                     const ALPHA_Number *x,
                                                                     const ALPHA_INT columns,
                                                                     const ALPHA_INT ldx,
                                                                     ALPHA_Number *y,
                                                                     const ALPHA_INT ldy) = {
    trsm_bsr_n_lo_row_plain,
    trsm_bsr_u_lo_row_plain,
    trsm_bsr_n_hi_row_plain,
    trsm_bsr_u_hi_row_plain,
    trsm_bsr_n_lo_col_plain,
    trsm_bsr_u_lo_col_plain,
    trsm_bsr_n_hi_col_plain,
    trsm_bsr_u_hi_col_plain,
    trsm_bsr_n_lo_row_trans_plain,
    trsm_bsr_u_lo_row_trans_plain,
    trsm_bsr_n_hi_row_trans_plain,
    trsm_bsr_u_hi_row_trans_plain,
    trsm_bsr_n_lo_col_trans_plain,
    trsm_bsr_u_lo_col_trans_plain,
    trsm_bsr_n_hi_col_trans_plain,
    trsm_bsr_u_hi_col_trans_plain,
#ifdef COMPLEX
    trsm_bsr_n_lo_row_conj_plain, 
    trsm_bsr_u_lo_row_conj_plain, 
    trsm_bsr_n_hi_row_conj_plain, 
    trsm_bsr_u_hi_row_conj_plain, 
    trsm_bsr_n_lo_col_conj_plain, 
    trsm_bsr_u_lo_col_conj_plain, 
    trsm_bsr_n_hi_col_conj_plain, 
    trsm_bsr_u_hi_col_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsm_bsr_diag_layout_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_BSR *A,
                                                        const ALPHA_Number *x,
                                                        const ALPHA_INT columns,
                                                        const ALPHA_INT ldx,
                                                        ALPHA_Number *y,
                                                        const ALPHA_INT ldy) = {
    diagsm_bsr_n_row_plain,
    diagsm_bsr_u_row_plain,
    diagsm_bsr_n_col_plain,
    diagsm_bsr_u_col_plain,
};

static alphasparse_status_t (*trsm_sky_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                     const ALPHA_SPMAT_SKY *A,
                                                                     const ALPHA_Number *x,
                                                                     const ALPHA_INT columns,
                                                                     const ALPHA_INT ldx,
                                                                     ALPHA_Number *y,
                                                                     const ALPHA_INT ldy) = {
    trsm_sky_n_lo_row_plain,
    trsm_sky_u_lo_row_plain,
    trsm_sky_n_hi_row_plain,
    trsm_sky_u_hi_row_plain,
    trsm_sky_n_lo_col_plain,
    trsm_sky_u_lo_col_plain,
    trsm_sky_n_hi_col_plain,
    trsm_sky_u_hi_col_plain,
    trsm_sky_n_lo_row_trans_plain,
    trsm_sky_u_lo_row_trans_plain,
    trsm_sky_n_hi_row_trans_plain,
    trsm_sky_u_hi_row_trans_plain,
    trsm_sky_n_lo_col_trans_plain,
    trsm_sky_u_lo_col_trans_plain,
    trsm_sky_n_hi_col_trans_plain,
    trsm_sky_u_hi_col_trans_plain,
#ifdef COMPLEX
    trsm_sky_n_lo_row_conj_plain, 
    trsm_sky_u_lo_row_conj_plain, 
    trsm_sky_n_hi_row_conj_plain, 
    trsm_sky_u_hi_row_conj_plain, 
    trsm_sky_n_lo_col_conj_plain, 
    trsm_sky_u_lo_col_conj_plain, 
    trsm_sky_n_hi_col_conj_plain, 
    trsm_sky_u_hi_col_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsm_sky_diag_layout_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_SKY *A,
                                                        const ALPHA_Number *x,
                                                        const ALPHA_INT columns,
                                                        const ALPHA_INT ldx,
                                                        ALPHA_Number *y,
                                                        const ALPHA_INT ldy) = {
    diagsm_sky_n_row_plain,
    diagsm_sky_u_row_plain,
    diagsm_sky_n_col_plain,
    diagsm_sky_u_col_plain,
};

static alphasparse_status_t (*trsm_dia_diag_fill_layout_operation_plain[])(const ALPHA_Number alpha,
                                                                     const ALPHA_SPMAT_DIA *A,
                                                                     const ALPHA_Number *x,
                                                                     const ALPHA_INT columns,
                                                                     const ALPHA_INT ldx,
                                                                     ALPHA_Number *y,
                                                                     const ALPHA_INT ldy) = {
    trsm_dia_n_lo_row_plain,
    trsm_dia_u_lo_row_plain,
    trsm_dia_n_hi_row_plain,
    trsm_dia_u_hi_row_plain,
    trsm_dia_n_lo_col_plain,
    trsm_dia_u_lo_col_plain,
    trsm_dia_n_hi_col_plain,
    trsm_dia_u_hi_col_plain,
    trsm_dia_n_lo_row_trans_plain,
    trsm_dia_u_lo_row_trans_plain,
    trsm_dia_n_hi_row_trans_plain,
    trsm_dia_u_hi_row_trans_plain,
    trsm_dia_n_lo_col_trans_plain,
    trsm_dia_u_lo_col_trans_plain,
    trsm_dia_n_hi_col_trans_plain,
    trsm_dia_u_hi_col_trans_plain,
#ifdef COMPLEX
    trsm_dia_n_lo_row_conj_plain, 
    trsm_dia_u_lo_row_conj_plain, 
    trsm_dia_n_hi_row_conj_plain, 
    trsm_dia_u_hi_row_conj_plain, 
    trsm_dia_n_lo_col_conj_plain, 
    trsm_dia_u_lo_col_conj_plain, 
    trsm_dia_n_hi_col_conj_plain, 
    trsm_dia_u_hi_col_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsm_dia_diag_layout_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_DIA *A,
                                                        const ALPHA_Number *x,
                                                        const ALPHA_INT columns,
                                                        const ALPHA_INT ldx,
                                                        ALPHA_Number *y,
                                                        const ALPHA_INT ldy) = {
    diagsm_dia_n_row_plain,
    diagsm_dia_u_row_plain,
    diagsm_dia_n_col_plain,
    diagsm_dia_u_col_plain,
};

alphasparse_status_t ONAME(const alphasparse_operation_t operation,
                                            const ALPHA_Number alpha,
                                            const alphasparse_matrix_t A,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                            const ALPHA_Number *x,
                                            const ALPHA_INT columns,
                                            const ALPHA_INT ldx,
                                            ALPHA_Number *y,
                                            const ALPHA_INT ldy)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
    if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif

    check_return(!check_equal_row_col(A),ALPHA_SPARSE_STATUS_INVALID_VALUE);

    if(A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsm_csr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsm_csr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsm_csr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsm_csc_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsm_csc_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsm_csc_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_COO)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsm_coo_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsm_coo_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsm_coo_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsm_coo_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_SKY)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsm_sky_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsm_sky_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsm_sky_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsm_bsr_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsm_bsr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsm_bsr_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_DIA)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsm_dia_diag_fill_layout_operation_plain[index4(operation, layout, descr.mode, descr.diag, ALPHA_SPARSE_LAYOUT_NUM, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsm_dia_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsm_dia_diag_layout_plain[index2(layout, descr.diag, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, y, ldy);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else
    {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
}
