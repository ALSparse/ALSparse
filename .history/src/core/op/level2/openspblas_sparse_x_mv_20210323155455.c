/**
 * @brief implement for openspblas_sparse_?_mv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "openspblas/util.h"
#include "openspblas/opt.h"
#include "openspblas/spapi.h"
#include "openspblas/kernel.h"


/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* gemv_csr          General matrics defined in csr storage format
* gemv_csr_trans    Transpose of general matrics defined in csr storage format
* gemv_csr_conj     Conjugate transpose of general matrics defined in csr storage format
* 
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* symv_csr_n_lo         symmetric matrics defined in csr, calculate lower triangle including diagal
* symv_csr_u_lo         symmetric matrics defined in csr, calculate lower triangle excluding diagal
* symv_csr_n_hi         symmetric matrics defined in csr, calculate higher triangle ixcluding diagal
* symv_csr_u_hi         symmetric matrics defined in csr, calculate higher triangle excluding diagal
* symv_csr_n_lo_conj    Conjugate transpose symmetric matrics defined in csr, calculate lower triangle including diagal
* symv_csr_u_lo_conj    Conjugate transpose symmetric matrics defined in csr, calculate lower triangle excluding diagal
* symv_csr_n_hi_conj    Conjugate transpose symmetric matrics defined in csr, calculate higher triangle ixcluding diagal
* symv_csr_u_hi_conj    Conjugate transpose symmetric matrics defined in csr, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* hemv_csr_n_lo         hermitian matrics defined in csr, calculate lower triangle including diagal
* hemv_csr_u_lo         hermitian matrics defined in csr, calculate lower triangle excluding diagal
* hemv_csr_n_hi         hermitian matrics defined in csr, calculate higher triangle excluding diagal
* hemv_csr_u_hi         hermitian matrics defined in csr, calculate higher triangle excluding diagal
* hemv_csr_n_lo_conj    Conjugate transpose hermitian matrics defined in csr, calculate lower triangle including diagal
* hemv_csr_u_lo_conj    Conjugate transpose hermitian matrics defined in csr, calculate lower triangle excluding diagal 
* hemv_csr_n_hi_conj    Conjugate transpose hermitian matrics defined in csr, calculate higher triangle excluding diagal
* hemv_csr_u_hi_conj    Conjugate transpose hermitian matrics defined in csr, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmv_csr_n_lo         triangular matrics defined in csr, calculate lower triangle including diagal
* trmv_csr_u_lo         triangular matrics defined in csr, calculate lower triangle excluding diagal
* trmv_csr_n_hi         triangular matrics defined in csr, calculate higher triangle excluding diagal
* trmv_csr_u_hi         triangular matrics defined in csr, calculate higher triangle excluding diagal
* trmv_csr_n_lo_conj    Conjugate transpose triangular matrics defined in csr, calculate lower triangle including diagal
* trmv_csr_u_lo_conj    Conjugate transpose triangular matrics defined in csr, calculate lower triangle excluding diagal
* trmv_csr_n_hi_conj    Conjugate transpose triangular matrics defined in csr, calculate higher triangle excluding diagal
* trmv_csr_u_hi_conj    Conjugate transpose triangular matrics defined in csr, calculate higher triangle excluding diagal
*
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmv_csr_n          diagonal matrics defined in csr, including diagal
* diagmv_csr_u          diagonal matrics defined in csr, excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

static openspblas_sparse_status_t (*diagmv_csr_diag[])(const OPENSPBLAS_Number alpha,
                                         const OPENSPBLAS_SPMAT_CSR *A,
                                         const OPENSPBLAS_Number *x,
                                         const OPENSPBLAS_Number beta,
                                         OPENSPBLAS_Number *y) = {
    diagmv_csr_n,
    diagmv_csr_u,
};

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* gemv_coo          General matrics defined in coo storage format
* gemv_coo_trans    Transpose of general matrics defined in coo storage format
* gemv_coo_conj     Conjugate transpose of general matrics defined in coo storage format
* 
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* symv_coo_n_lo         symmetric matrics defined in coo, calculate lower triangle including diagal
* symv_coo_u_lo         symmetric matrics defined in coo, calculate lower triangle excluding diagal
* symv_coo_n_hi         symmetric matrics defined in coo, calculate higher triangle ixcluding diagal
* symv_coo_u_hi         symmetric matrics defined in coo, calculate higher triangle excluding diagal
* symv_coo_n_lo_conj    Conjugate transpose symmetric matrics defined in coo, calculate lower triangle including diagal
* symv_coo_u_lo_conj    Conjugate transpose symmetric matrics defined in coo, calculate lower triangle excluding diagal
* symv_coo_n_hi_conj    Conjugate transpose symmetric matrics defined in coo, calculate higher triangle ixcluding diagal
* symv_coo_u_hi_conj    Conjugate transpose symmetric matrics defined in coo, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* hemv_coo_n_lo         hermitian matrics defined in coo, calculate lower triangle including diagal
* hemv_coo_u_lo         hermitian matrics defined in coo, calculate lower triangle excluding diagal
* hemv_coo_n_hi         hermitian matrics defined in coo, calculate higher triangle excluding diagal
* hemv_coo_u_hi         hermitian matrics defined in coo, calculate higher triangle excluding diagal
* hemv_coo_n_lo_conj    Conjugate transpose hermitian matrics defined in coo, calculate lower triangle including diagal
* hemv_coo_u_lo_conj    Conjugate transpose hermitian matrics defined in coo, calculate lower triangle excluding diagal 
* hemv_coo_n_hi_conj    Conjugate transpose hermitian matrics defined in coo, calculate higher triangle excluding diagal
* hemv_coo_u_hi_conj    Conjugate transpose hermitian matrics defined in coo, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmv_coo_n_lo         triangular matrics defined in coo, calculate lower triangle including diagal
* trmv_coo_u_lo         triangular matrics defined in coo, calculate lower triangle excluding diagal
* trmv_coo_n_hi         triangular matrics defined in coo, calculate higher triangle excluding diagal
* trmv_coo_u_hi         triangular matrics defined in coo, calculate higher triangle excluding diagal
* trmv_coo_n_lo_conj    Conjugate transpose triangular matrics defined in coo, calculate lower triangle including diagal
* trmv_coo_u_lo_conj    Conjugate transpose triangular matrics defined in coo, calculate lower triangle excluding diagal
* trmv_coo_n_hi_conj    Conjugate transpose triangular matrics defined in coo, calculate higher triangle excluding diagal
* trmv_coo_u_hi_conj    Conjugate transpose triangular matrics defined in coo, calculate higher triangle excluding diagal
*
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmv_coo_n          diagonal matrics defined in coo, including diagal
* diagmv_coo_u          diagonal matrics defined in coo, excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

static openspblas_sparse_status_t (*diagmv_coo_diag[])(const OPENSPBLAS_Number alpha,
                                         const OPENSPBLAS_SPMAT_COO *A,
                                         const OPENSPBLAS_Number *x,
                                         const OPENSPBLAS_Number beta,
                                         OPENSPBLAS_Number *y) = {
    diagmv_coo_n,
    diagmv_coo_u,
};

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* gemv_csc          General matrics defined in csc storage format
* gemv_csc_trans    Transpose of general matrics defined in csc storage format
* gemv_csc_conj     Conjugate transpose of general matrics defined in csc storage format
* 
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* symv_csc_n_lo         symmetric matrics defined in csc, calculate lower triangle including diagal
* symv_csc_u_lo         symmetric matrics defined in csc, calculate lower triangle excluding diagal
* symv_csc_n_hi         symmetric matrics defined in csc, calculate higher triangle ixcluding diagal
* symv_csc_u_hi         symmetric matrics defined in csc, calculate higher triangle excluding diagal
* symv_csc_n_lo_conj    Conjugate transpose symmetric matrics defined in csc, calculate lower triangle including diagal
* symv_csc_u_lo_conj    Conjugate transpose symmetric matrics defined in csc, calculate lower triangle excluding diagal
* symv_csc_n_hi_conj    Conjugate transpose symmetric matrics defined in csc, calculate higher triangle ixcluding diagal
* symv_csc_u_hi_conj    Conjugate transpose symmetric matrics defined in csc, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* hemv_csc_n_lo         hermitian matrics defined in csc, calculate lower triangle including diagal
* hemv_csc_u_lo         hermitian matrics defined in csc, calculate lower triangle excluding diagal
* hemv_csc_n_hi         hermitian matrics defined in csc, calculate higher triangle excluding diagal
* hemv_csc_u_hi         hermitian matrics defined in csc, calculate higher triangle excluding diagal
* hemv_csc_n_lo_conj    Conjugate transpose hermitian matrics defined in csc, calculate lower triangle including diagal
* hemv_csc_u_lo_conj    Conjugate transpose hermitian matrics defined in csc, calculate lower triangle excluding diagal 
* hemv_csc_n_hi_conj    Conjugate transpose hermitian matrics defined in csc, calculate higher triangle excluding diagal
* hemv_csc_u_hi_conj    Conjugate transpose hermitian matrics defined in csc, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmv_csc_n_lo         triangular matrics defined in csc, calculate lower triangle including diagal
* trmv_csc_u_lo         triangular matrics defined in csc, calculate lower triangle excluding diagal
* trmv_csc_n_hi         triangular matrics defined in csc, calculate higher triangle excluding diagal
* trmv_csc_u_hi         triangular matrics defined in csc, calculate higher triangle excluding diagal
* trmv_csc_n_lo_conj    Conjugate transpose triangular matrics defined in csc, calculate lower triangle including diagal
* trmv_csc_u_lo_conj    Conjugate transpose triangular matrics defined in csc, calculate lower triangle excluding diagal
* trmv_csc_n_hi_conj    Conjugate transpose triangular matrics defined in csc, calculate higher triangle excluding diagal
* trmv_csc_u_hi_conj    Conjugate transpose triangular matrics defined in csc, calculate higher triangle excluding diagal
*
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmv_csc_n          diagonal matrics defined in csc, including diagal
* diagmv_csc_u          diagonal matrics defined in csc, excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

static openspblas_sparse_status_t (*diagmv_csc_diag[])(const OPENSPBLAS_Number alpha,
                                         const OPENSPBLAS_SPMAT_CSC *A,
                                         const OPENSPBLAS_Number *x,
                                         const OPENSPBLAS_Number beta,
                                         OPENSPBLAS_Number *y) = {
    diagmv_csc_n,
    diagmv_csc_u,
};

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* gemv_sky          General matrics defined in sky storage format
* gemv_sky_trans    Transpose of general matrics defined in sky storage format
* gemv_sky_conj     Conjugate transpose of general matrics defined in sky storage format
* 
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* symv_sky_n_lo         symmetric matrics defined in sky, calculate lower triangle including diagal
* symv_sky_u_lo         symmetric matrics defined in sky, calculate lower triangle excluding diagal
* symv_sky_n_hi         symmetric matrics defined in sky, calculate higher triangle ixcluding diagal
* symv_sky_u_hi         symmetric matrics defined in sky, calculate higher triangle excluding diagal
* symv_sky_n_lo_conj    Conjugate transpose symmetric matrics defined in sky, calculate lower triangle including diagal
* symv_sky_u_lo_conj    Conjugate transpose symmetric matrics defined in sky, calculate lower triangle excluding diagal
* symv_sky_n_hi_conj    Conjugate transpose symmetric matrics defined in sky, calculate higher triangle ixcluding diagal
* symv_sky_u_hi_conj    Conjugate transpose symmetric matrics defined in sky, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* hemv_sky_n_lo         hermitian matrics defined in sky, calculate lower triangle including diagal
* hemv_sky_u_lo         hermitian matrics defined in sky, calculate lower triangle excluding diagal
* hemv_sky_n_hi         hermitian matrics defined in sky, calculate higher triangle excluding diagal
* hemv_sky_u_hi         hermitian matrics defined in sky, calculate higher triangle excluding diagal
* hemv_sky_n_lo_conj    Conjugate transpose hermitian matrics defined in sky, calculate lower triangle including diagal
* hemv_sky_u_lo_conj    Conjugate transpose hermitian matrics defined in sky, calculate lower triangle excluding diagal 
* hemv_sky_n_hi_conj    Conjugate transpose hermitian matrics defined in sky, calculate higher triangle excluding diagal
* hemv_sky_u_hi_conj    Conjugate transpose hermitian matrics defined in sky, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmv_sky_n_lo         triangular matrics defined in sky, calculate lower triangle including diagal
* trmv_sky_u_lo         triangular matrics defined in sky, calculate lower triangle excluding diagal
* trmv_sky_n_hi         triangular matrics defined in sky, calculate higher triangle excluding diagal
* trmv_sky_u_hi         triangular matrics defined in sky, calculate higher triangle excluding diagal
* trmv_sky_n_lo_conj    Conjugate transpose triangular matrics defined in sky, calculate lower triangle including diagal
* trmv_sky_u_lo_conj    Conjugate transpose triangular matrics defined in sky, calculate lower triangle excluding diagal
* trmv_sky_n_hi_conj    Conjugate transpose triangular matrics defined in sky, calculate higher triangle excluding diagal
* trmv_sky_u_hi_conj    Conjugate transpose triangular matrics defined in sky, calculate higher triangle excluding diagal
*
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmv_sky_n          diagonal matrics defined in sky, including diagal
* diagmv_sky_u          diagonal matrics defined in sky, excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

static openspblas_sparse_status_t (*diagmv_sky_diag[])(const OPENSPBLAS_Number alpha,
                                                 const OPENSPBLAS_SPMAT_SKY *A,
                                                 const OPENSPBLAS_Number *x,
                                                 const OPENSPBLAS_Number beta,
                                                 OPENSPBLAS_Number *y) = {
    diagmv_sky_n,
    diagmv_sky_u,
};

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* gemv_bsr          General matrics defined in bsr storage format
* gemv_bsr_trans    Transpose of general matrics defined in bsr storage format
* gemv_bsr_conj     Conjugate transpose of general matrics defined in bsr storage format
* 
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* symv_bsr_n_lo         symmetric matrics defined in bsr, calculate lower triangle including diagal
* symv_bsr_u_lo         symmetric matrics defined in bsr, calculate lower triangle excluding diagal
* symv_bsr_n_hi         symmetric matrics defined in bsr, calculate higher triangle ixcluding diagal
* symv_bsr_u_hi         symmetric matrics defined in bsr, calculate higher triangle excluding diagal
* symv_bsr_n_lo_conj    Conjugate transpose symmetric matrics defined in bsr, calculate lower triangle including diagal
* symv_bsr_u_lo_conj    Conjugate transpose symmetric matrics defined in bsr, calculate lower triangle excluding diagal
* symv_bsr_n_hi_conj    Conjugate transpose symmetric matrics defined in bsr, calculate higher triangle ixcluding diagal
* symv_bsr_u_hi_conj    Conjugate transpose symmetric matrics defined in bsr, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/


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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* hemv_bsr_n_lo         hermitian matrics defined in bsr, calculate lower triangle including diagal
* hemv_bsr_u_lo         hermitian matrics defined in bsr, calculate lower triangle excluding diagal
* hemv_bsr_n_hi         hermitian matrics defined in bsr, calculate higher triangle excluding diagal
* hemv_bsr_u_hi         hermitian matrics defined in bsr, calculate higher triangle excluding diagal
* hemv_bsr_n_lo_conj    Conjugate transpose hermitian matrics defined in bsr, calculate lower triangle including diagal
* hemv_bsr_u_lo_conj    Conjugate transpose hermitian matrics defined in bsr, calculate lower triangle excluding diagal 
* hemv_bsr_n_hi_conj    Conjugate transpose hermitian matrics defined in bsr, calculate higher triangle excluding diagal
* hemv_bsr_u_hi_conj    Conjugate transpose hermitian matrics defined in bsr, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/


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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmv_bsr_n_lo         triangular matrics defined in bsr, calculate lower triangle including diagal
* trmv_bsr_u_lo         triangular matrics defined in bsr, calculate lower triangle excluding diagal
* trmv_bsr_n_hi         triangular matrics defined in bsr, calculate higher triangle excluding diagal
* trmv_bsr_u_hi         triangular matrics defined in bsr, calculate higher triangle excluding diagal
* trmv_bsr_n_lo_conj    Conjugate transpose triangular matrics defined in bsr, calculate lower triangle including diagal
* trmv_bsr_u_lo_conj    Conjugate transpose triangular matrics defined in bsr, calculate lower triangle excluding diagal
* trmv_bsr_n_hi_conj    Conjugate transpose triangular matrics defined in bsr, calculate higher triangle excluding diagal
* trmv_bsr_u_hi_conj    Conjugate transpose triangular matrics defined in bsr, calculate higher triangle excluding diagal
*
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmv_dia_n          diagonal matrics defined in dia, including diagal
* diagmv_dia_u          diagonal matrics defined in dia, excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

static openspblas_sparse_status_t (*diagmv_bsr_diag[])(const OPENSPBLAS_Number alpha,
                                                 const OPENSPBLAS_SPMAT_BSR *A,
                                                 const OPENSPBLAS_Number *x,
                                                 const OPENSPBLAS_Number beta,
                                                 OPENSPBLAS_Number *y) = {
    diagmv_bsr_n,
    diagmv_bsr_u,
};

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* gemv_dia          General matrics defined in dia storage format
* gemv_dia_trans    Transpose of general matrics defined in dia storage format
* gemv_dia_conj     Conjugate transpose of general matrics defined in dia storage format
* 
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* symv_dia_n_lo         symmetric matrics defined in dia, calculate lower triangle including diagal
* symv_dia_u_lo         symmetric matrics defined in dia, calculate lower triangle excluding diagal
* symv_dia_n_hi         symmetric matrics defined in dia, calculate higher triangle ixcluding diagal
* symv_dia_u_hi         symmetric matrics defined in dia, calculate higher triangle excluding diagal
* symv_dia_n_lo_conj    Conjugate transpose symmetric matrics defined in dia, calculate lower triangle including diagal
* symv_dia_u_lo_conj    Conjugate transpose symmetric matrics defined in dia, calculate lower triangle excluding diagal
* symv_dia_n_hi_conj    Conjugate transpose symmetric matrics defined in dia, calculate higher triangle ixcluding diagal
* symv_dia_u_hi_conj    Conjugate transpose symmetric matrics defined in dia, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a vector
*
* details:
* hemv_dia_n_lo         hermitian matrics defined in dia, calculate lower triangle including diagal
* hemv_dia_u_lo         hermitian matrics defined in dia, calculate lower triangle excluding diagal
* hemv_dia_n_hi         hermitian matrics defined in dia, calculate higher triangle excluding diagal
* hemv_dia_u_hi         hermitian matrics defined in dia, calculate higher triangle excluding diagal
* hemv_dia_n_lo_conj    Conjugate transpose hermitian matrics defined in dia, calculate lower triangle including diagal
* hemv_dia_u_lo_conj    Conjugate transpose hermitian matrics defined in dia, calculate lower triangle excluding diagal 
* hemv_dia_n_hi_conj    Conjugate transpose hermitian matrics defined in dia, calculate higher triangle excluding diagal
* hemv_dia_u_hi_conj    Conjugate transpose hermitian matrics defined in dia, calculate higher triangle excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
*
* input:
* alpha     a scalor value alpha
* beta      a scalor value beta
* x         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of columns of matrix A
* y         a dense matrix, stored as an array, if the matrix A is not transposed, 
            the length is at least the number of rowss of matrix A
* A         sparse matrix(k*m)
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmv_dia_n_lo         triangular matrics defined in dia, calculate lower triangle including diagal
* trmv_dia_u_lo         triangular matrics defined in dia, calculate lower triangle excluding diagal
* trmv_dia_n_hi         triangular matrics defined in dia, calculate higher triangle excluding diagal
* trmv_dia_u_hi         triangular matrics defined in dia, calculate higher triangle excluding diagal
* trmv_dia_n_lo_conj    Conjugate transpose triangular matrics defined in dia, calculate lower triangle including diagal
* trmv_dia_u_lo_conj    Conjugate transpose triangular matrics defined in dia, calculate lower triangle excluding diagal
* trmv_dia_n_hi_conj    Conjugate transpose triangular matrics defined in dia, calculate higher triangle excluding diagal
* trmv_dia_u_hi_conj    Conjugate transpose triangular matrics defined in dia, calculate higher triangle excluding diagal
*
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

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

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmv_dia_n          diagonal matrics defined in dia, including diagal
* diagmv_dia_u          diagonal matrics defined in dia, excluding diagal
* y := alpha * op(A) * x + beta * y
*
* alpha     a scalor value
* beta      a scalor value
* x         a dense matrix
* y         a dense matrix
* op(A)     Data structure of sparse matrix
* op: op(A) = A
*     op(A) = AT
*     op(A) = AH
*
* input:
* alpha     a scalor value
* beta      a scalor value
* x         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of x     ldx        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of x     columns    ldx
* y         Dense matrix, stored as an array, at least rows*cols
                    row major   column major
    row value of y     ldy        columns of A(op(A) = A);rows of A(op(A) = AT)
    col value of y     columns    ldy
* mat       sparse matrix
* columns   Number of columns of dense matrix y
* ldx       main dimension of the matrix x
* ldy       main dimension of the matrix y
* 
* output:
* y         a dense matrix
*
*/

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
