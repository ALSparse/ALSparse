/**
 * @brief implement for alphasparse_?_trsv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "alphasparse/spdef.h"

static alphasparse_status_t (*trsv_csr_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_CSR *A,
                                                        const ALPHA_Number *x,
                                                        ALPHA_Number *y) = {
    trsv_csr_n_lo_plain,
    trsv_csr_u_lo_plain,
    trsv_csr_n_hi_plain,
    trsv_csr_u_hi_plain,
    trsv_csr_n_lo_trans_plain,
    trsv_csr_u_lo_trans_plain,
    trsv_csr_n_hi_trans_plain,
    trsv_csr_u_hi_trans_plain,
#ifdef COMPLEX
    trsv_csr_n_lo_conj_plain, 
    trsv_csr_u_lo_conj_plain, 
    trsv_csr_n_hi_conj_plain, 
    trsv_csr_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsv_csr_diag_plain[])(const ALPHA_Number alpha,
                                           const ALPHA_SPMAT_CSR *A,
                                           const ALPHA_Number *x,
                                           ALPHA_Number *y) = {
    diagsv_csr_n_plain,
    diagsv_csr_u_plain,
};

static alphasparse_status_t (*trsv_coo_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_COO *A,
                                                        const ALPHA_Number *x,
                                                        ALPHA_Number *y) = {
    trsv_coo_n_lo_plain,
    trsv_coo_u_lo_plain,
    trsv_coo_n_hi_plain,
    trsv_coo_u_hi_plain,
    trsv_coo_n_lo_trans_plain,
    trsv_coo_u_lo_trans_plain,
    trsv_coo_n_hi_trans_plain,
    trsv_coo_u_hi_trans_plain,
#ifdef COMPLEX
    trsv_coo_n_lo_conj_plain, 
    trsv_coo_u_lo_conj_plain, 
    trsv_coo_n_hi_conj_plain, 
    trsv_coo_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsv_coo_diag_plain[])(const ALPHA_Number alpha,
                                           const ALPHA_SPMAT_COO *A,
                                           const ALPHA_Number *x,
                                           ALPHA_Number *y) = {
    diagsv_coo_n_plain,
    diagsv_coo_u_plain,
};

static alphasparse_status_t (*trsv_bsr_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_BSR *A,
                                                        const ALPHA_Number *x,
                                                        ALPHA_Number *y) = {
    trsv_bsr_n_lo_plain,
    trsv_bsr_u_lo_plain,
    trsv_bsr_n_hi_plain,
    trsv_bsr_u_hi_plain,
    trsv_bsr_n_lo_trans_plain,
    trsv_bsr_u_lo_trans_plain,
    trsv_bsr_n_hi_trans_plain,
    trsv_bsr_u_hi_trans_plain,
#ifdef COMPLEX
    trsv_bsr_n_lo_conj_plain, 
    trsv_bsr_u_lo_conj_plain, 
    trsv_bsr_n_hi_conj_plain, 
    trsv_bsr_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsv_bsr_diag_plain[])(const ALPHA_Number alpha,
                                           const ALPHA_SPMAT_BSR *A,
                                           const ALPHA_Number *x,
                                           ALPHA_Number *y) = {
    diagsv_bsr_n_plain,
    diagsv_bsr_u_plain,
};

static alphasparse_status_t (*trsv_csc_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_CSC *A,
                                                        const ALPHA_Number *x,
                                                        ALPHA_Number *y) = {
    trsv_csc_n_lo_plain,
    trsv_csc_u_lo_plain,
    trsv_csc_n_hi_plain,
    trsv_csc_u_hi_plain,
    trsv_csc_n_lo_trans_plain,
    trsv_csc_u_lo_trans_plain,
    trsv_csc_n_hi_trans_plain,
    trsv_csc_u_hi_trans_plain,
#ifdef COMPLEX
    trsv_csc_n_lo_conj_plain, 
    trsv_csc_u_lo_conj_plain, 
    trsv_csc_n_hi_conj_plain, 
    trsv_csc_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsv_csc_diag_plain[])(const ALPHA_Number alpha,
                                           const ALPHA_SPMAT_CSC *A,
                                           const ALPHA_Number *x,
                                           ALPHA_Number *y) = {
    diagsv_csc_n_plain,
    diagsv_csc_u_plain,
};

static alphasparse_status_t (*trsv_sky_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_SKY *A,
                                                        const ALPHA_Number *x,
                                                        ALPHA_Number *y) = {
    trsv_sky_n_lo_plain,
    trsv_sky_u_lo_plain,
    trsv_sky_n_hi_plain,
    trsv_sky_u_hi_plain,
    trsv_sky_n_lo_trans_plain,
    trsv_sky_u_lo_trans_plain,
    trsv_sky_n_hi_trans_plain,
    trsv_sky_u_hi_trans_plain,
#ifdef COMPLEX
    trsv_sky_n_lo_conj_plain, 
    trsv_sky_u_lo_conj_plain, 
    trsv_sky_n_hi_conj_plain, 
    trsv_sky_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsv_sky_diag_plain[])(const ALPHA_Number alpha,
                                           const ALPHA_SPMAT_SKY *A,
                                           const ALPHA_Number *x,
                                           ALPHA_Number *y) = {
    diagsv_sky_n_plain,
    diagsv_sky_u_plain,
};

static alphasparse_status_t (*trsv_dia_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                        const ALPHA_SPMAT_DIA *A,
                                                        const ALPHA_Number *x,
                                                        ALPHA_Number *y) = {
    trsv_dia_n_lo_plain,
    trsv_dia_u_lo_plain,
    trsv_dia_n_hi_plain,
    trsv_dia_u_hi_plain,
    trsv_dia_n_lo_trans_plain,
    trsv_dia_u_lo_trans_plain,
    trsv_dia_n_hi_trans_plain,
    trsv_dia_u_hi_trans_plain,
#ifdef COMPLEX
    trsv_dia_n_lo_conj_plain, 
    trsv_dia_u_lo_conj_plain, 
    trsv_dia_n_hi_conj_plain, 
    trsv_dia_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagsv_dia_diag_plain[])(const ALPHA_Number alpha,
                                           const ALPHA_SPMAT_DIA *A,
                                           const ALPHA_Number *x,
                                           ALPHA_Number *y) = {
    diagsv_dia_n_plain,
    diagsv_dia_u_plain,
};

alphasparse_status_t ONAME(const alphasparse_operation_t operation, const ALPHA_Number alpha, const alphasparse_matrix_t A, const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const ALPHA_Number *x, ALPHA_Number *y)
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
            check_null_return(trsv_csr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsv_csr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsv_csr_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsv_csr_diag_plain[descr.diag](alpha, A->mat, x, y);
        }else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsv_bsr_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsv_bsr_diag_plain[descr.diag](alpha, A->mat, x, y);
        }else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }       
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_COO)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsv_coo_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsv_coo_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsv_coo_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsv_coo_diag_plain[descr.diag](alpha, A->mat, x, y);
        }else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }       
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsv_csc_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsv_csc_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsv_csc_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsv_csc_diag_plain[descr.diag](alpha, A->mat, x, y);
        }else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }       
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_SKY)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsv_sky_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsv_sky_diag_plain[descr.diag](alpha, A->mat, x, y);
        }else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }       
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_DIA)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trsv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trsv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagsv_dia_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagsv_dia_diag_plain[descr.diag](alpha, A->mat, x, y);
        }else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }       
    }
    else
    {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }  
}
