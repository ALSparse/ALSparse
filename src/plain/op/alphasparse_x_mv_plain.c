/**
 * @brief implement for alphasparse_?_mv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"

static alphasparse_status_t (*gemv_csr_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_CSR *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    gemv_csr_plain,
    gemv_csr_trans_plain,
#ifdef COMPLEX
    gemv_csr_conj_plain, 
#endif
};

static alphasparse_status_t (*symv_csr_diag_fill_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_CSR *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    symv_csr_n_lo_plain,
    symv_csr_u_lo_plain,
    symv_csr_n_hi_plain,
    symv_csr_u_hi_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_csr_n_lo_conj_plain,
    symv_csr_u_lo_conj_plain,
    symv_csr_n_hi_conj_plain,
    symv_csr_u_hi_conj_plain,
#endif
};

static alphasparse_status_t (*hermv_csr_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                              const ALPHA_SPMAT_CSR *A,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y) = {
#ifndef COMPLEX    
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermv_csr_n_lo_plain,
    hermv_csr_u_lo_plain,
    hermv_csr_n_hi_plain,
    hermv_csr_u_hi_plain,
    hermv_csr_n_lo_trans_plain,
    hermv_csr_u_lo_trans_plain,
    hermv_csr_n_hi_trans_plain,
    hermv_csr_u_hi_trans_plain
#endif
};

static alphasparse_status_t (*trmv_csr_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                              const ALPHA_SPMAT_CSR *A,
                                                              const ALPHA_Number *x,
                                                              const ALPHA_Number beta,
                                                              ALPHA_Number *y) = {
    trmv_csr_n_lo_plain,
    trmv_csr_u_lo_plain,
    trmv_csr_n_hi_plain,
    trmv_csr_u_hi_plain,
    trmv_csr_n_lo_trans_plain,
    trmv_csr_u_lo_trans_plain,
    trmv_csr_n_hi_trans_plain,
    trmv_csr_u_hi_trans_plain,
#ifdef COMPLEX
    trmv_csr_n_lo_conj_plain, 
    trmv_csr_u_lo_conj_plain, 
    trmv_csr_n_hi_conj_plain, 
    trmv_csr_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagmv_csr_diag_plain[])(const ALPHA_Number alpha,
                                                 const ALPHA_SPMAT_CSR *A,
                                                 const ALPHA_Number *x,
                                                 const ALPHA_Number beta,
                                                 ALPHA_Number *y) = {
    diagmv_csr_n_plain,
    diagmv_csr_u_plain,
};

static alphasparse_status_t (*gemv_coo_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_COO *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    gemv_coo_plain,
    gemv_coo_trans_plain,
#ifdef COMPLEX
    gemv_coo_conj_plain, 
#endif
};

static alphasparse_status_t (*symv_coo_diag_fill_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_COO *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    symv_coo_n_lo_plain,
    symv_coo_u_lo_plain,
    symv_coo_n_hi_plain,
    symv_coo_u_hi_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_coo_n_lo_conj_plain,
    symv_coo_u_lo_conj_plain,
    symv_coo_n_hi_conj_plain,
    symv_coo_u_hi_conj_plain,
#endif
};

static alphasparse_status_t (*hermv_coo_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                              const ALPHA_SPMAT_COO *A,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y) = {
#ifndef COMPLEX    
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermv_coo_n_lo_plain,
    hermv_coo_u_lo_plain,
    hermv_coo_n_hi_plain,
    hermv_coo_u_hi_plain,
    hermv_coo_n_lo_trans_plain,
    hermv_coo_u_lo_trans_plain,
    hermv_coo_n_hi_trans_plain,
    hermv_coo_u_hi_trans_plain
#endif
};

static alphasparse_status_t (*trmv_coo_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                              const ALPHA_SPMAT_COO *A,
                                                              const ALPHA_Number *x,
                                                              const ALPHA_Number beta,
                                                              ALPHA_Number *y) = {
    trmv_coo_n_lo_plain,
    trmv_coo_u_lo_plain,
    trmv_coo_n_hi_plain,
    trmv_coo_u_hi_plain,
    trmv_coo_n_lo_trans_plain,
    trmv_coo_u_lo_trans_plain,
    trmv_coo_n_hi_trans_plain,
    trmv_coo_u_hi_trans_plain,
#ifdef COMPLEX
    trmv_coo_n_lo_conj_plain, 
    trmv_coo_u_lo_conj_plain, 
    trmv_coo_n_hi_conj_plain, 
    trmv_coo_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagmv_coo_diag_plain[])(const ALPHA_Number alpha,
                                                 const ALPHA_SPMAT_COO *A,
                                                 const ALPHA_Number *x,
                                                 const ALPHA_Number beta,
                                                 ALPHA_Number *y) = {
    diagmv_coo_n_plain,
    diagmv_coo_u_plain,
};

static alphasparse_status_t (*gemv_csc_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_CSC *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    gemv_csc_plain,
    gemv_csc_trans_plain,
#ifdef COMPLEX
    gemv_csc_conj_plain, 
#endif
};

static alphasparse_status_t (*symv_csc_diag_fill_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_CSC *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    symv_csc_n_lo_plain,
    symv_csc_u_lo_plain,
    symv_csc_n_hi_plain,
    symv_csc_u_hi_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_csc_n_lo_conj_plain,
    symv_csc_u_lo_conj_plain,
    symv_csc_n_hi_conj_plain,
    symv_csc_u_hi_conj_plain,
#endif
};

static alphasparse_status_t (*hermv_csc_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                              const ALPHA_SPMAT_CSC *A,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y) = {
#ifndef COMPLEX    
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermv_csc_n_lo_plain,
    hermv_csc_u_lo_plain,
    hermv_csc_n_hi_plain,
    hermv_csc_u_hi_plain,
    hermv_csc_n_lo_trans_plain,
    hermv_csc_u_lo_trans_plain,
    hermv_csc_n_hi_trans_plain,
    hermv_csc_u_hi_trans_plain
#endif
};

static alphasparse_status_t (*trmv_csc_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                              const ALPHA_SPMAT_CSC *A,
                                                              const ALPHA_Number *x,
                                                              const ALPHA_Number beta,
                                                              ALPHA_Number *y) = {
    trmv_csc_n_lo_plain,
    trmv_csc_u_lo_plain,
    trmv_csc_n_hi_plain,
    trmv_csc_u_hi_plain,
    trmv_csc_n_lo_trans_plain,
    trmv_csc_u_lo_trans_plain,
    trmv_csc_n_hi_trans_plain,
    trmv_csc_u_hi_trans_plain,
#ifdef COMPLEX
    trmv_csc_n_lo_conj_plain, 
    trmv_csc_u_lo_conj_plain, 
    trmv_csc_n_hi_conj_plain, 
    trmv_csc_u_hi_conj_plain, 
#endif
};

static alphasparse_status_t (*diagmv_csc_diag_plain[])(const ALPHA_Number alpha,
                                                 const ALPHA_SPMAT_CSC *A,
                                                 const ALPHA_Number *x,
                                                 const ALPHA_Number beta,
                                                 ALPHA_Number *y) = {
    diagmv_csc_n_plain,
    diagmv_csc_u_plain,
};

static alphasparse_status_t (*gemv_bsr_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_BSR *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    gemv_bsr_plain,
    gemv_bsr_trans_plain,
#ifdef COMPLEX
    gemv_bsr_conj_plain, 
#endif
};

static alphasparse_status_t (*symv_bsr_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_BSR *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    symv_bsr_n_lo_plain,
    symv_bsr_u_lo_plain,
    symv_bsr_n_hi_plain,
    symv_bsr_u_hi_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_bsr_n_lo_conj_plain,
    symv_bsr_u_lo_conj_plain,
    symv_bsr_n_hi_conj_plain,
    symv_bsr_u_hi_conj_plain,
#endif
};

static alphasparse_status_t (*hermv_bsr_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                              const ALPHA_SPMAT_BSR *A,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y) = {
#ifndef COMPLEX    
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermv_bsr_n_lo_plain,
    hermv_bsr_u_lo_plain,
    hermv_bsr_n_hi_plain,
    hermv_bsr_u_hi_plain,
    hermv_bsr_n_lo_trans_plain,
    hermv_bsr_u_lo_trans_plain,
    hermv_bsr_n_hi_trans_plain,
    hermv_bsr_u_hi_trans_plain
#endif
};

static alphasparse_status_t (*trmv_bsr_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                              const ALPHA_SPMAT_BSR *A,
                                                              const ALPHA_Number *x,
                                                              const ALPHA_Number beta,
                                                              ALPHA_Number *y) = {
    trmv_bsr_n_lo_plain,
    trmv_bsr_u_lo_plain,
    trmv_bsr_n_hi_plain,
    trmv_bsr_u_hi_plain,
    trmv_bsr_n_lo_trans_plain,
    trmv_bsr_u_lo_trans_plain,
    trmv_bsr_n_hi_trans_plain,
    trmv_bsr_u_hi_trans_plain,
#ifdef COMPLEX
    trmv_bsr_n_lo_conj_plain, 
    trmv_bsr_u_lo_conj_plain, 
    trmv_bsr_n_hi_conj_plain, 
    trmv_bsr_u_hi_conj_plain, 
#endif
};
static alphasparse_status_t (*diagmv_bsr_diag_plain[])(const ALPHA_Number alpha,
                                                 const ALPHA_SPMAT_BSR *A,
                                                 const ALPHA_Number *x,
                                                 const ALPHA_Number beta,
                                                 ALPHA_Number *y) = {
    diagmv_bsr_n_plain,
    diagmv_bsr_u_plain,
};

static alphasparse_status_t (*gemv_sky_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_SKY *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    gemv_sky_plain,
    gemv_sky_trans_plain,
#ifdef COMPLEX
    NULL, 
#endif
};
static alphasparse_status_t (*symv_sky_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_SKY *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    symv_sky_n_lo_plain,
    symv_sky_u_lo_plain,
    symv_sky_n_hi_plain,
    symv_sky_u_hi_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_sky_n_lo_conj_plain,
    symv_sky_u_lo_conj_plain,
    symv_sky_n_hi_conj_plain,
    symv_sky_u_hi_conj_plain,
#endif
};
static alphasparse_status_t (*hermv_sky_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                              const ALPHA_SPMAT_SKY *A,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y) = {
#ifndef COMPLEX    
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermv_sky_n_lo_plain,
    hermv_sky_u_lo_plain,
    hermv_sky_n_hi_plain,
    hermv_sky_u_hi_plain,
    hermv_sky_n_lo_trans_plain,
    hermv_sky_u_lo_trans_plain,
    hermv_sky_n_hi_trans_plain,
    hermv_sky_u_hi_trans_plain
#endif
};
static alphasparse_status_t (*trmv_sky_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                              const ALPHA_SPMAT_SKY *A,
                                                              const ALPHA_Number *x,
                                                              const ALPHA_Number beta,
                                                              ALPHA_Number *y) = {
    trmv_sky_n_lo_plain,
    trmv_sky_u_lo_plain,
    trmv_sky_n_hi_plain,
    trmv_sky_u_hi_plain,
    trmv_sky_n_lo_trans_plain,
    trmv_sky_u_lo_trans_plain,
    trmv_sky_n_hi_trans_plain,
    trmv_sky_u_hi_trans_plain,
#ifdef COMPLEX
    trmv_sky_n_lo_conj_plain, 
    trmv_sky_u_lo_conj_plain, 
    trmv_sky_n_hi_conj_plain, 
    trmv_sky_u_hi_conj_plain, 
#endif
};
static alphasparse_status_t (*diagmv_sky_diag_plain[])(const ALPHA_Number alpha,
                                                 const ALPHA_SPMAT_SKY *A,
                                                 const ALPHA_Number *x,
                                                 const ALPHA_Number beta,
                                                 ALPHA_Number *y) = {
    diagmv_sky_n_plain,
    diagmv_sky_u_plain,
};

static alphasparse_status_t (*gemv_dia_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_DIA *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    gemv_dia_plain,
    gemv_dia_trans_plain,
#ifdef COMPLEX
    gemv_dia_conj_plain, 
#endif
};

static alphasparse_status_t (*symv_dia_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                    const ALPHA_SPMAT_DIA *A,
                                                    const ALPHA_Number *x,
                                                    const ALPHA_Number beta,
                                                    ALPHA_Number *y) = {
    symv_dia_n_lo_plain,
    symv_dia_u_lo_plain,
    symv_dia_n_hi_plain,
    symv_dia_u_hi_plain,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_dia_n_lo_conj_plain,
    symv_dia_u_lo_conj_plain,
    symv_dia_n_hi_conj_plain,
    symv_dia_u_hi_conj_plain,
#endif
};
static alphasparse_status_t (*hermv_dia_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                              const ALPHA_SPMAT_DIA *A,
                                              const ALPHA_Number *x,
                                              const ALPHA_Number beta,
                                              ALPHA_Number *y) = {
#ifndef COMPLEX    
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#else
    hermv_dia_n_lo_plain,
    hermv_dia_u_lo_plain,
    hermv_dia_n_hi_plain,
    hermv_dia_u_hi_plain,
    hermv_dia_n_lo_trans_plain,
    hermv_dia_u_lo_trans_plain,
    hermv_dia_n_hi_trans_plain,
    hermv_dia_u_hi_trans_plain
#endif
};
static alphasparse_status_t (*trmv_dia_diag_fill_operation_plain[])(const ALPHA_Number alpha,
                                                              const ALPHA_SPMAT_DIA *A,
                                                              const ALPHA_Number *x,
                                                              const ALPHA_Number beta,
                                                              ALPHA_Number *y) = {
    trmv_dia_n_lo_plain,
    trmv_dia_u_lo_plain,
    trmv_dia_n_hi_plain,
    trmv_dia_u_hi_plain,
    trmv_dia_n_lo_trans_plain,
    trmv_dia_u_lo_trans_plain,
    trmv_dia_n_hi_trans_plain,
    trmv_dia_u_hi_trans_plain,
#ifdef COMPLEX
    trmv_dia_n_lo_conj_plain, 
    trmv_dia_u_lo_conj_plain, 
    trmv_dia_n_hi_conj_plain, 
    trmv_dia_u_hi_conj_plain, 
#endif
};
static alphasparse_status_t (*diagmv_dia_diag_plain[])(const ALPHA_Number alpha,
                                                 const ALPHA_SPMAT_DIA *A,
                                                 const ALPHA_Number *x,
                                                 const ALPHA_Number beta,
                                                 ALPHA_Number *y) = {
    diagmv_dia_n_plain,
    diagmv_dia_u_plain,
};

alphasparse_status_t ONAME(const alphasparse_operation_t operation,
                          const ALPHA_Number alpha,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                          const ALPHA_Number *x,
                          const ALPHA_Number beta,
                          ALPHA_Number *y)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
    if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif

    if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC || descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        // Check if it is a square matrix 
        check_return(!check_equal_row_col(A),ALPHA_SPARSE_STATUS_INVALID_VALUE);

    if (A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_csr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_csr_operation_plain[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_csr_diag_fill_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_csr_diag_fill_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_csr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_csr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_csr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_csr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_csr_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_csr_diag_plain[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == ALPHA_SPARSE_FORMAT_COO)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_coo_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_coo_operation_plain[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_coo_diag_fill_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_coo_diag_fill_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_coo_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return ( hermv_coo_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha,A->mat,x,beta,y) );
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_coo_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_coo_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_coo_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_coo_diag_plain[descr.diag](alpha, A->mat, x, beta, y);
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
            check_null_return(gemv_csc_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_csc_operation_plain[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_csc_diag_fill_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_csc_diag_fill_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_csc_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_csc_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_csc_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_csc_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_csc_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_csc_diag_plain[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_bsr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_bsr_operation_plain[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if(descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return ( hermv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha,A->mat,x,beta,y) );
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_bsr_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_bsr_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_bsr_diag_plain[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_sky_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_sky_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_sky_diag_plain[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
    {
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_dia_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_dia_operation_plain[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if ( descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_dia_diag_fill_operation_plain[index3(operation, descr.mode, descr.diag, ALPHA_SPARSE_FILL_MODE_NUM, ALPHA_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_dia_diag_plain[descr.diag], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_dia_diag_plain[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}
