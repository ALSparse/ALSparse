/**
 * @brief implement for openspblas_sparse_?_mv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "openspblas/util.h"
#include "openspblas/opt.h"
#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

static openspblas_sparse_status_t (*gemv_csr_operation[])(const OPENSPBLAS_Number alpha,
                                            const OPENSPBLAS_SPMAT_CSR *A,
                                            const OPENSPBLAS_Number *x,
                                            const OPENSPBLAS_Number beta,
                                            OPENSPBLAS_Number *y) = {
    gemv_csr,
    gemv_csr_trans,
#ifdef COMPLEX
    gemv_csr_conj, 
#endif
};

static openspblas_sparse_status_t (*symv_csr_diag_fill[])(const OPENSPBLAS_Number alpha,
                                            const OPENSPBLAS_SPMAT_CSR *A,
                                            const OPENSPBLAS_Number *x,
                                            const OPENSPBLAS_Number beta,
                                            OPENSPBLAS_Number *y) = {
    symv_csr_n_lo,
    symv_csr_u_lo,
    symv_csr_n_hi,
    symv_csr_u_hi,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_csr_n_lo_conj,
    symv_csr_u_lo_conj,
    symv_csr_n_hi_conj,
    symv_csr_u_hi_conj,
#endif
};

static openspblas_sparse_status_t (*hermv_csr_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                              const OPENSPBLAS_SPMAT_CSR *A,
                                              const OPENSPBLAS_Number *x,
                                              const OPENSPBLAS_Number beta,
                                              OPENSPBLAS_Number *y) = {
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
    hermv_csr_n_lo,
    hermv_csr_u_lo,
    hermv_csr_n_hi,
    hermv_csr_u_hi,
    hermv_csr_n_lo_trans,
    hermv_csr_u_lo_trans,
    hermv_csr_n_hi_trans,
    hermv_csr_u_hi_trans
#endif
};

static openspblas_sparse_status_t (*trmv_csr_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_CSR *A,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y) = {
    trmv_csr_n_lo,
    trmv_csr_u_lo,
    trmv_csr_n_hi,
    trmv_csr_u_hi,
    trmv_csr_n_lo_trans,
    trmv_csr_u_lo_trans,
    trmv_csr_n_hi_trans,
    trmv_csr_u_hi_trans,
#ifdef COMPLEX
    trmv_csr_n_lo_conj, 
    trmv_csr_u_lo_conj, 
    trmv_csr_n_hi_conj, 
    trmv_csr_u_hi_conj, 
#endif
};

static openspblas_sparse_status_t (*diagmv_csr_diag[])(const OPENSPBLAS_Number alpha,
                                         const OPENSPBLAS_SPMAT_CSR *A,
                                         const OPENSPBLAS_Number *x,
                                         const OPENSPBLAS_Number beta,
                                         OPENSPBLAS_Number *y) = {
    diagmv_csr_n,
    diagmv_csr_u,
};

static openspblas_sparse_status_t (*gemv_coo_operation[])(const OPENSPBLAS_Number alpha,
                                            const OPENSPBLAS_SPMAT_COO *A,
                                            const OPENSPBLAS_Number *x,
                                            const OPENSPBLAS_Number beta,
                                            OPENSPBLAS_Number *y) = {
    gemv_coo,
    gemv_coo_trans,
#ifdef COMPLEX
    gemv_coo_conj, 
#endif
};

static openspblas_sparse_status_t (*symv_coo_diag_fill[])(const OPENSPBLAS_Number alpha,
                                            const OPENSPBLAS_SPMAT_COO *A,
                                            const OPENSPBLAS_Number *x,
                                            const OPENSPBLAS_Number beta,
                                            OPENSPBLAS_Number *y) = {
    symv_coo_n_lo,
    symv_coo_u_lo,
    symv_coo_n_hi,
    symv_coo_u_hi,
    NULL, // padding
    NULL, // padding
    NULL, // padding
    NULL, // padding
#ifdef COMPLEX
    symv_coo_n_lo_conj,
    symv_coo_u_lo_conj,
    symv_coo_n_hi_conj,
    symv_coo_u_hi_conj,
#endif 
};

static openspblas_sparse_status_t (*hermv_coo_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                              const OPENSPBLAS_SPMAT_COO *A,
                                              const OPENSPBLAS_Number *x,
                                              const OPENSPBLAS_Number beta,
                                              OPENSPBLAS_Number *y) = {
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
    hermv_coo_n_lo,
    hermv_coo_u_lo,
    hermv_coo_n_hi,
    hermv_coo_u_hi,
    hermv_coo_n_lo_trans,
    hermv_coo_u_lo_trans,
    hermv_coo_n_hi_trans,
    hermv_coo_u_hi_trans
#endif
};

static openspblas_sparse_status_t (*trmv_coo_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_COO *A,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y) = {
    trmv_coo_n_lo,
    trmv_coo_u_lo,
    trmv_coo_n_hi,
    trmv_coo_u_hi,
    trmv_coo_n_lo_trans,
    trmv_coo_u_lo_trans,
    trmv_coo_n_hi_trans,
    trmv_coo_u_hi_trans,
#ifdef COMPLEX
    trmv_coo_n_lo_conj, 
    trmv_coo_u_lo_conj, 
    trmv_coo_n_hi_conj, 
    trmv_coo_u_hi_conj, 
#endif
};

static openspblas_sparse_status_t (*diagmv_coo_diag[])(const OPENSPBLAS_Number alpha,
                                         const OPENSPBLAS_SPMAT_COO *A,
                                         const OPENSPBLAS_Number *x,
                                         const OPENSPBLAS_Number beta,
                                         OPENSPBLAS_Number *y) = {
    diagmv_coo_n,
    diagmv_coo_u,
};

static openspblas_sparse_status_t (*gemv_csc_operation[])(const OPENSPBLAS_Number alpha,
                                            const OPENSPBLAS_SPMAT_CSC *A,
                                            const OPENSPBLAS_Number *x,
                                            const OPENSPBLAS_Number beta,
                                            OPENSPBLAS_Number *y) = {
    gemv_csc,
    gemv_csc_trans,
#ifdef COMPLEX
    gemv_csc_conj, 
#endif
};

static openspblas_sparse_status_t (*symv_csc_diag_fill[])(const OPENSPBLAS_Number alpha,
                                            const OPENSPBLAS_SPMAT_CSC *A,
                                            const OPENSPBLAS_Number *x,
                                            const OPENSPBLAS_Number beta,
                                            OPENSPBLAS_Number *y) = {
    symv_csc_n_lo,
    symv_csc_u_lo,
    symv_csc_n_hi,
    symv_csc_u_hi,
    NULL, // padding
    NULL, // padding
    NULL, // padding
    NULL, // padding
#ifdef COMPLEX
    symv_csc_n_lo_conj,
    symv_csc_u_lo_conj,
    symv_csc_n_hi_conj,
    symv_csc_u_hi_conj,
#endif
};

static openspblas_sparse_status_t (*hermv_csc_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                              const OPENSPBLAS_SPMAT_CSC *A,
                                              const OPENSPBLAS_Number *x,
                                              const OPENSPBLAS_Number beta,
                                              OPENSPBLAS_Number *y) = {
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
    hermv_csc_n_lo,
    hermv_csc_u_lo,
    hermv_csc_n_hi,
    hermv_csc_u_hi,
    hermv_csc_n_lo_trans,
    hermv_csc_u_lo_trans,
    hermv_csc_n_hi_trans,
    hermv_csc_u_hi_trans
#endif
};

static openspblas_sparse_status_t (*trmv_csc_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_CSC *A,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y) = {
    trmv_csc_n_lo,
    trmv_csc_u_lo,
    trmv_csc_n_hi,
    trmv_csc_u_hi,
    trmv_csc_n_lo_trans,
    trmv_csc_u_lo_trans,
    trmv_csc_n_hi_trans,
    trmv_csc_u_hi_trans,
#ifdef COMPLEX
    trmv_csc_n_lo_conj, 
    trmv_csc_u_lo_conj, 
    trmv_csc_n_hi_conj, 
    trmv_csc_u_hi_conj, 
#endif
};

static openspblas_sparse_status_t (*diagmv_csc_diag[])(const OPENSPBLAS_Number alpha,
                                         const OPENSPBLAS_SPMAT_CSC *A,
                                         const OPENSPBLAS_Number *x,
                                         const OPENSPBLAS_Number beta,
                                         OPENSPBLAS_Number *y) = {
    diagmv_csc_n,
    diagmv_csc_u,
};

static openspblas_sparse_status_t (*gemv_sky_operation[])(const OPENSPBLAS_Number alpha,
                                                    const OPENSPBLAS_SPMAT_SKY *A,
                                                    const OPENSPBLAS_Number *x,
                                                    const OPENSPBLAS_Number beta,
                                                    OPENSPBLAS_Number *y) = {
    gemv_sky,
    gemv_sky_trans,
#ifdef COMPLEX
    NULL, 
#endif
};
static openspblas_sparse_status_t (*symv_sky_diag_fill[])(const OPENSPBLAS_Number alpha,
                                                    const OPENSPBLAS_SPMAT_SKY *A,
                                                    const OPENSPBLAS_Number *x,
                                                    const OPENSPBLAS_Number beta,
                                                    OPENSPBLAS_Number *y) = {
    symv_sky_n_lo,
    symv_sky_u_lo,
    symv_sky_n_hi,
    symv_sky_u_hi,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_sky_n_lo_conj,
    symv_sky_u_lo_conj,
    symv_sky_n_hi_conj,
    symv_sky_u_hi_conj,
#endif
};

static openspblas_sparse_status_t (*hermv_sky_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                              const OPENSPBLAS_SPMAT_SKY *A,
                                              const OPENSPBLAS_Number *x,
                                              const OPENSPBLAS_Number beta,
                                              OPENSPBLAS_Number *y) = {
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
    hermv_sky_n_lo,
    hermv_sky_u_lo,
    hermv_sky_n_hi,
    hermv_sky_u_hi,
    hermv_sky_n_lo_trans,
    hermv_sky_u_lo_trans,
    hermv_sky_n_hi_trans,
    hermv_sky_u_hi_trans
#endif
};

static openspblas_sparse_status_t (*trmv_sky_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                                              const OPENSPBLAS_SPMAT_SKY *A,
                                                              const OPENSPBLAS_Number *x,
                                                              const OPENSPBLAS_Number beta,
                                                              OPENSPBLAS_Number *y) = {
    trmv_sky_n_lo,
    trmv_sky_u_lo,
    trmv_sky_n_hi,
    trmv_sky_u_hi,
    trmv_sky_n_lo_trans,
    trmv_sky_u_lo_trans,
    trmv_sky_n_hi_trans,
    trmv_sky_u_hi_trans,
#ifdef COMPLEX
    trmv_sky_n_lo_conj, 
    trmv_sky_u_lo_conj, 
    trmv_sky_n_hi_conj, 
    trmv_sky_u_hi_conj, 
#endif
};
static openspblas_sparse_status_t (*diagmv_sky_diag[])(const OPENSPBLAS_Number alpha,
                                                 const OPENSPBLAS_SPMAT_SKY *A,
                                                 const OPENSPBLAS_Number *x,
                                                 const OPENSPBLAS_Number beta,
                                                 OPENSPBLAS_Number *y) = {
    diagmv_sky_n,
    diagmv_sky_u,
};

static openspblas_sparse_status_t (*gemv_bsr_operation[])(const OPENSPBLAS_Number alpha,
                                                    const OPENSPBLAS_SPMAT_BSR *A,
                                                    const OPENSPBLAS_Number *x,
                                                    const OPENSPBLAS_Number beta,
                                                    OPENSPBLAS_Number *y) = {
    gemv_bsr,
    gemv_bsr_trans,
#ifdef COMPLEX
    gemv_bsr_conj, 
#endif
};
static openspblas_sparse_status_t (*symv_bsr_diag_fill[])(const OPENSPBLAS_Number alpha,
                                                    const OPENSPBLAS_SPMAT_BSR *A,
                                                    const OPENSPBLAS_Number *x,
                                                    const OPENSPBLAS_Number beta,
                                                    OPENSPBLAS_Number *y) = {
    symv_bsr_n_lo,
    symv_bsr_u_lo,
    symv_bsr_n_hi,
    symv_bsr_u_hi,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_bsr_n_lo_conj,
    symv_bsr_u_lo_conj,
    symv_bsr_n_hi_conj,
    symv_bsr_u_hi_conj,
#endif
};

static openspblas_sparse_status_t (*hermv_bsr_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                              const OPENSPBLAS_SPMAT_BSR *A,
                                              const OPENSPBLAS_Number *x,
                                              const OPENSPBLAS_Number beta,
                                              OPENSPBLAS_Number *y) = {
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
    hermv_bsr_n_lo,
    hermv_bsr_u_lo,
    hermv_bsr_n_hi,
    hermv_bsr_u_hi,
    hermv_bsr_n_lo_trans,
    hermv_bsr_u_lo_trans,
    hermv_bsr_n_hi_trans,
    hermv_bsr_u_hi_trans
#endif
};

static openspblas_sparse_status_t (*trmv_bsr_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                                              const OPENSPBLAS_SPMAT_BSR *A,
                                                              const OPENSPBLAS_Number *x,
                                                              const OPENSPBLAS_Number beta,
                                                              OPENSPBLAS_Number *y) = {
    trmv_bsr_n_lo,
    trmv_bsr_u_lo,
    trmv_bsr_n_hi,
    trmv_bsr_u_hi,
    trmv_bsr_n_lo_trans,
    trmv_bsr_u_lo_trans,
    trmv_bsr_n_hi_trans,
    trmv_bsr_u_hi_trans,
#ifdef COMPLEX
    trmv_bsr_n_lo_conj, 
    trmv_bsr_u_lo_conj, 
    trmv_bsr_n_hi_conj, 
    trmv_bsr_u_hi_conj, 
#endif
};

static openspblas_sparse_status_t (*diagmv_bsr_diag[])(const OPENSPBLAS_Number alpha,
                                                 const OPENSPBLAS_SPMAT_BSR *A,
                                                 const OPENSPBLAS_Number *x,
                                                 const OPENSPBLAS_Number beta,
                                                 OPENSPBLAS_Number *y) = {
    diagmv_bsr_n,
    diagmv_bsr_u,
};

static openspblas_sparse_status_t (*gemv_dia_operation[])(const OPENSPBLAS_Number alpha,
                                                    const OPENSPBLAS_SPMAT_DIA *A,
                                                    const OPENSPBLAS_Number *x,
                                                    const OPENSPBLAS_Number beta,
                                                    OPENSPBLAS_Number *y) = {
    gemv_dia,
    gemv_dia_trans,
#ifdef COMPLEX
    gemv_dia_conj, 
#endif
};

static openspblas_sparse_status_t (*symv_dia_diag_fill[])(const OPENSPBLAS_Number alpha,
                                                    const OPENSPBLAS_SPMAT_DIA *A,
                                                    const OPENSPBLAS_Number *x,
                                                    const OPENSPBLAS_Number beta,
                                                    OPENSPBLAS_Number *y) = {
    symv_dia_n_lo,
    symv_dia_u_lo,
    symv_dia_n_hi,
    symv_dia_u_hi,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symv_dia_n_lo_conj,
    symv_dia_u_lo_conj,
    symv_dia_n_hi_conj,
    symv_dia_u_hi_conj,
#endif
};

static openspblas_sparse_status_t (*hermv_dia_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                              const OPENSPBLAS_SPMAT_DIA *A,
                                              const OPENSPBLAS_Number *x,
                                              const OPENSPBLAS_Number beta,
                                              OPENSPBLAS_Number *y) = {
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
    hermv_dia_n_lo,
    hermv_dia_u_lo,
    hermv_dia_n_hi,
    hermv_dia_u_hi,
    hermv_dia_n_lo_trans,
    hermv_dia_u_lo_trans,
    hermv_dia_n_hi_trans,
    hermv_dia_u_hi_trans
#endif
};

static openspblas_sparse_status_t (*trmv_dia_diag_fill_operation[])(const OPENSPBLAS_Number alpha,
                                                              const OPENSPBLAS_SPMAT_DIA *A,
                                                              const OPENSPBLAS_Number *x,
                                                              const OPENSPBLAS_Number beta,
                                                              OPENSPBLAS_Number *y) = {
    trmv_dia_n_lo,
    trmv_dia_u_lo,
    trmv_dia_n_hi,
    trmv_dia_u_hi,
    trmv_dia_n_lo_trans,
    trmv_dia_u_lo_trans,
    trmv_dia_n_hi_trans,
    trmv_dia_u_hi_trans,
#ifdef COMPLEX
    trmv_dia_n_lo_conj, 
    trmv_dia_u_lo_conj, 
    trmv_dia_n_hi_conj, 
    trmv_dia_u_hi_conj, 
#endif
};
static openspblas_sparse_status_t (*diagmv_dia_diag[])(const OPENSPBLAS_Number alpha,
                                                 const OPENSPBLAS_SPMAT_DIA *A,
                                                 const OPENSPBLAS_Number *x,
                                                 const OPENSPBLAS_Number beta,
                                                 OPENSPBLAS_Number *y) = {
    diagmv_dia_n,
    diagmv_dia_u,
};

openspblas_sparse_status_t ONAME(const openspblas_sparse_operation_t operation,
                          const OPENSPBLAS_Number alpha,
                          const openspblas_sparse_matrix_t A,
                          const struct openspblas_matrix_descr descr, /* openspblas_sparse_matrix_type_t + openspblas_sparse_fill_mode_t + openspblas_sparse_diag_type_t */
                          const OPENSPBLAS_Number *x,
                          const OPENSPBLAS_Number beta,
                          OPENSPBLAS_Number *y)
{
    check_null_return(A, OPENSPBLAS_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(A->mat, OPENSPBLAS_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, OPENSPBLAS_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, OPENSPBLAS_SPARSE_STATUS_NOT_INITIALIZED);
    check_return(A->datatype != OPENSPBLAS_SPARSE_DATATYPE, OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
    if(operation == OPENSPBLAS_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
#endif

    if(descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC || descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        // check if it is a square matrix 
        check_return(!check_equal_row_col(A),OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE);


    if (A->format == OPENSPBLAS_SPARSE_FORMAT_CSR)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_csr_operation[operation], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_csr_operation[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_csr_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_csr_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_csr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_csr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_csr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_csr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_csr_diag[descr.diag], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_csr_diag[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == OPENSPBLAS_SPARSE_FORMAT_COO)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_coo_operation[operation], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_coo_operation[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_coo_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_coo_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if(descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_coo_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return ( hermv_coo_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha,A->mat,x,beta,y) );
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_coo_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_coo_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_coo_diag[descr.diag], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_coo_diag[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == OPENSPBLAS_SPARSE_FORMAT_CSC)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_csc_operation[operation], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_csc_operation[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_csc_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_csc_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_csc_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_csc_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_csc_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_csc_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_csc_diag[descr.diag], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_csc_diag[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == OPENSPBLAS_SPARSE_FORMAT_SKY)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_return(operation != OPENSPBLAS_SPARSE_OPERATION_NON_TRANSPOSE, OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            check_null_return(symv_sky_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_sky_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_sky_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_sky_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_sky_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_sky_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_sky_diag[descr.diag], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_sky_diag[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == OPENSPBLAS_SPARSE_FORMAT_BSR)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_bsr_operation[operation], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_bsr_operation[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_bsr_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_bsr_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if(descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_bsr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return ( hermv_bsr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha,A->mat,x,beta,y) );
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_bsr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_bsr_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_bsr_diag[descr.diag], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_bsr_diag[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == OPENSPBLAS_SPARSE_FORMAT_DIA)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemv_dia_operation[operation], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemv_dia_operation[operation](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symv_dia_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symv_dia_diag_fill[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermv_dia_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermv_dia_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmv_dia_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmv_dia_diag_fill_operation[index3(operation, descr.mode, descr.diag, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, beta, y);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmv_dia_diag[descr.diag], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmv_dia_diag[descr.diag](alpha, A->mat, x, beta, y);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else
    {
        return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
    }
}
