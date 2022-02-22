#include "openspblas/util.h"
#include "openspblas/opt.h"
#include "openspblas/spapi.h"
#include "openspblas/kernel.h"
#include "openspblas/spdef.h"

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* gemm_csr_row          General row major matrics defined in csr storage format
* gemm_csr_col          General column major matrics defined in csr storage format 
* gemm_csr_row_trans    Transpose of general column major matrics defined in csr storage format
* gemm_csr_col_trans    Transpose of general column major matrics defined in csr storage format
* gemm_csr_row_conj     Conjugate transpose of general column major matrics defined in csr storage format
* gemm_csr_col_conj     Conjugate transpose of general column major matrics defined in csr storage format 
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

static openspblas_sparse_status_t (*gemm_csr_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_CSR *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    gemm_csr_row,
    gemm_csr_col,
    gemm_csr_row_trans,
    gemm_csr_col_trans,
#ifdef COMPLEX
    gemm_csr_row_conj,
    gemm_csr_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* symm_csr_n_lo_row         symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row         symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col         symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col         symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*symm_csr_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_CSR *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    symm_csr_n_lo_row,
    symm_csr_u_lo_row,
    symm_csr_n_hi_row,
    symm_csr_u_hi_row,
    symm_csr_n_lo_col,
    symm_csr_u_lo_col,
    symm_csr_n_hi_col,
    symm_csr_u_hi_col,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_csr_n_lo_row_conj,
    symm_csr_u_lo_row_conj,
    symm_csr_n_hi_row_conj,
    symm_csr_u_hi_row_conj,
    symm_csr_n_lo_col_conj,
    symm_csr_u_lo_col_conj,
    symm_csr_n_hi_col_conj,
    symm_csr_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* hemm_csr_n_lo_row         hermitian row major matrics defined in csr, calculate lower triangle including diagal
* hemm_csr_u_lo_row         hermitian row major matrics defined in csr, calculate lower triangle excluding diagal
* hemm_csr_n_hi_row         hermitian row major matrics defined in csr, calculate higher triangle excluding diagal
* hemm_csr_u_hi_row         hermitian row major matrics defined in csr, calculate higher triangle excluding diagal
* hemm_csr_n_lo_col         hermitian column major matrics defined in csr, calculate lower triangle including diagal
* hemm_csr_u_lo_col         hermitian column major matrics defined in csr, calculate lower triangle excluding diagal 
* hemm_csr_n_hi_col         hermitian column major matrics defined in csr, calculate higher triangle excluding diagal
* hemm_csr_u_hi_col         hermitian column major matrics defined in csr, calculate higher triangle excluding diagal
* hemm_csr_n_lo_row_conj    Conjugate transpose hermitian row major matrics defined in csr, calculate lower triangle including diagal
* hemm_csr_u_lo_row_conj    Conjugate transpose hermitian row major matrics defined in csr, calculate lower triangle excluding diagal
* hemm_csr_n_hi_row_conj    Conjugate transpose hermitian row major matrics defined in csr, calculate higher triangle excluding diagal
* hemm_csr_u_hi_row_conj    Conjugate transpose hermitian row major matrics defined in csr, calculate higher triangle excluding diagal
* hemm_csr_n_lo_col_conj    Conjugate transpose hermitian column major matrics defined in csr, calculate lower triangle including diagal
* hemm_csr_u_lo_col_conj    Conjugate transpose hermitian column major matrics defined in csr, calculate lower triangle excluding diagal 
* hemm_csr_n_hi_col_conj    Conjugate transpose hermitian column major matrics defined in csr, calculate higher triangle excluding diagal
* hemm_csr_u_hi_col_conj    Conjugate transpose hermitian column major matrics defined in csr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*hermm_csr_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                     const OPENSPBLAS_SPMAT_CSR *mat,
                                                     const OPENSPBLAS_Number *x,
                                                     const OPENSPBLAS_INT columns,
                                                     const OPENSPBLAS_INT ldx,
                                                     const OPENSPBLAS_Number beta,
                                                     OPENSPBLAS_Number *y,
                                                     const OPENSPBLAS_INT ldy) = {
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
    hermm_csr_n_lo_row,
    hermm_csr_u_lo_row,
    hermm_csr_n_hi_row,
    hermm_csr_u_hi_row,
    hermm_csr_n_lo_col,
    hermm_csr_u_lo_col,
    hermm_csr_n_hi_col,
    hermm_csr_u_hi_col,
    
    hermm_csr_n_lo_row_trans,
    hermm_csr_u_lo_row_trans,
    hermm_csr_n_hi_row_trans,
    hermm_csr_u_hi_row_trans,
    hermm_csr_n_lo_col_trans,
    hermm_csr_u_lo_col_trans,
    hermm_csr_n_hi_col_trans,
    hermm_csr_u_hi_col_trans,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmm_csr_n_lo_row         triangular row major matrics defined in csr, calculate lower triangle including diagal
* trmm_csr_u_lo_row         triangular row major matrics defined in csr, calculate lower triangle excluding diagal
* trmm_csr_n_hi_row         triangular row major matrics defined in csr, calculate higher triangle excluding diagal
* trmm_csr_u_hi_row         triangular row major matrics defined in csr, calculate higher triangle excluding diagal
* trmm_csr_n_lo_col         triangular column major matrics defined in csr, calculate lower triangle including diagal
* trmm_csr_u_lo_col         triangular column major matrics defined in csr, calculate lower triangle excluding diagal 
* trmm_csr_n_hi_col         triangular column major matrics defined in csr, calculate higher triangle excluding diagal
* trmm_csr_u_hi_col         triangular column major matrics defined in csr, calculate higher triangle excluding diagal
* trmm_csr_n_lo_row_conj    Conjugate transpose triangular row major matrics defined in csr, calculate lower triangle including diagal
* trmm_csr_u_lo_row_conj    Conjugate transpose triangular row major matrics defined in csr, calculate lower triangle excluding diagal
* trmm_csr_n_hi_row_conj    Conjugate transpose triangular row major matrics defined in csr, calculate higher triangle excluding diagal
* trmm_csr_u_hi_row_conj    Conjugate transpose triangular row major matrics defined in csr, calculate higher triangle excluding diagal
* trmm_csr_n_lo_col_conj    Conjugate transpose triangular column major matrics defined in csr, calculate lower triangle including diagal
* trmm_csr_u_lo_col_conj    Conjugate transpose triangular column major matrics defined in csr, calculate lower triangle excluding diagal 
* trmm_csr_n_hi_col_conj    Conjugate transpose triangular column major matrics defined in csr, calculate higher triangle excluding diagal
* trmm_csr_u_hi_col_conj    Conjugate transpose triangular column major matrics defined in csr, calculate higher triangle excluding diagal
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

static openspblas_sparse_status_t (*trmm_csr_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                                   const OPENSPBLAS_SPMAT_CSR *mat,
                                                                   const OPENSPBLAS_Number *x,
                                                                   const OPENSPBLAS_INT columns,
                                                                   const OPENSPBLAS_INT ldx,
                                                                   const OPENSPBLAS_Number beta,
                                                                   OPENSPBLAS_Number *y,
                                                                   const OPENSPBLAS_INT ldy) = {
    trmm_csr_n_lo_row,
    trmm_csr_u_lo_row,
    trmm_csr_n_hi_row,
    trmm_csr_u_hi_row,
    trmm_csr_n_lo_col,
    trmm_csr_u_lo_col,
    trmm_csr_n_hi_col,
    trmm_csr_u_hi_col,

    trmm_csr_n_lo_row_trans,
    trmm_csr_u_lo_row_trans,
    trmm_csr_n_hi_row_trans,
    trmm_csr_u_hi_row_trans,
    trmm_csr_n_lo_col_trans,
    trmm_csr_u_lo_col_trans,
    trmm_csr_n_hi_col_trans,
    trmm_csr_u_hi_col_trans,
#ifdef COMPLEX
    trmm_csr_n_lo_row_conj,
    trmm_csr_u_lo_row_conj,
    trmm_csr_n_hi_row_conj,
    trmm_csr_u_hi_row_conj,
    trmm_csr_n_lo_col_conj,
    trmm_csr_u_lo_col_conj,
    trmm_csr_n_hi_col_conj,
    trmm_csr_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmm_csr_n_row          diagonal row major matrics defined in csr, including diagal
* diagmm_csr_u_row          diagonal row major matrics defined in csr, excluding diagal
* diagmm_csr_n_col          diagonal column major matrics defined in csr, including diagal
* diagmm_csr_u_col          diagonal column major matrics defined in csr, excluding diagal
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


static openspblas_sparse_status_t (*diagmm_csr_diag_layout[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_CSR *mat,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_INT columns,
                                                      const OPENSPBLAS_INT ldx,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y,
                                                      const OPENSPBLAS_INT ldy) = {
    diagmm_csr_n_row,
    diagmm_csr_u_row,
    diagmm_csr_n_col,
    diagmm_csr_u_col,
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* gemm_coo_row          General row major matrics defined in coo storage format
* gemm_coo_col          General column major matrics defined in coo storage format 
* gemm_coo_row_trans    Transpose of general column major matrics defined in coo storage format
* gemm_coo_col_trans    Transpose of general column major matrics defined in coo storage format
* gemm_coo_row_conj     Conjugate transpose of general column major matrics defined in coo storage format
* gemm_coo_col_conj     Conjugate transpose of general column major matrics defined in coo storage format 
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

static openspblas_sparse_status_t (*gemm_coo_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_COO *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    gemm_coo_row,
    gemm_coo_col,
    gemm_coo_row_trans,
    gemm_coo_col_trans,
#ifdef COMPLEX
    gemm_coo_row_conj,
    gemm_coo_col_conj,
#endif 
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* symm_coo_n_lo_row         symmetric row major matrics defined in coo, calculate lower triangle including diagal
* symm_coo_u_lo_row         symmetric row major matrics defined in coo, calculate lower triangle excluding diagal
* symm_coo_n_hi_row         symmetric row major matrics defined in coo, calculate higher triangle excluding diagal
* symm_coo_u_hi_row         symmetric row major matrics defined in coo, calculate higher triangle excluding diagal
* symm_coo_n_lo_col         symmetric column major matrics defined in coo, calculate lower triangle including diagal
* symm_coo_u_lo_col         symmetric column major matrics defined in coo, calculate lower triangle excluding diagal 
* symm_coo_n_hi_col         symmetric column major matrics defined in coo, calculate higher triangle excluding diagal
* symm_coo_u_hi_col         symmetric column major matrics defined in coo, calculate higher triangle excluding diagal
* symm_coo_n_lo_row_conj    Conjugate transpose symmetric row major matrics defined in coo, calculate lower triangle including diagal
* symm_coo_u_lo_row_conj    Conjugate transpose symmetric row major matrics defined in coo, calculate lower triangle excluding diagal
* symm_coo_n_hi_row_conj    Conjugate transpose symmetric row major matrics defined in coo, calculate higher triangle excluding diagal
* symm_coo_u_hi_row_conj    Conjugate transpose symmetric row major matrics defined in coo, calculate higher triangle excluding diagal
* symm_coo_n_lo_col_conj    Conjugate transpose symmetric column major matrics defined in coo, calculate lower triangle including diagal
* symm_coo_u_lo_col_conj    Conjugate transpose symmetric column major matrics defined in coo, calculate lower triangle excluding diagal 
* symm_coo_n_hi_col_conj    Conjugate transpose symmetric column major matrics defined in coo, calculate higher triangle excluding diagal
* symm_coo_u_hi_col_conj    Conjugate transpose symmetric column major matrics defined in coo, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*symm_coo_diag_fill_layout[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_COO *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    symm_coo_n_lo_row,
    symm_coo_u_lo_row,
    symm_coo_n_hi_row,
    symm_coo_u_hi_row,
    symm_coo_n_lo_col,
    symm_coo_u_lo_col,
    symm_coo_n_hi_col,
    symm_coo_u_hi_col,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_coo_n_lo_row_conj,
    symm_coo_u_lo_row_conj,
    symm_coo_n_hi_row_conj,
    symm_coo_u_hi_row_conj,
    symm_coo_n_lo_col_conj,
    symm_coo_u_lo_col_conj,
    symm_coo_n_hi_col_conj,
    symm_coo_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* hemm_coo_n_lo_row         hermitian row major matrics defined in coo, calculate lower triangle including diagal
* hemm_coo_u_lo_row         hermitian row major matrics defined in coo, calculate lower triangle excluding diagal
* hemm_coo_n_hi_row         hermitian row major matrics defined in coo, calculate higher triangle excluding diagal
* hemm_coo_u_hi_row         hermitian row major matrics defined in coo, calculate higher triangle excluding diagal
* hemm_coo_n_lo_col         hermitian column major matrics defined in coo, calculate lower triangle including diagal
* hemm_coo_u_lo_col         hermitian column major matrics defined in coo, calculate lower triangle excluding diagal 
* hemm_coo_n_hi_col         hermitian column major matrics defined in coo, calculate higher triangle excluding diagal
* hemm_coo_u_hi_col         hermitian column major matrics defined in coo, calculate higher triangle excluding diagal
* hemm_coo_n_lo_row_conj    Conjugate transpose hermitian row major matrics defined in coo, calculate lower triangle including diagal
* hemm_coo_u_lo_row_conj    Conjugate transpose hermitian row major matrics defined in coo, calculate lower triangle excluding diagal
* hemm_coo_n_hi_row_conj    Conjugate transpose hermitian row major matrics defined in coo, calculate higher triangle excluding diagal
* hemm_coo_u_hi_row_conj    Conjugate transpose hermitian row major matrics defined in coo, calculate higher triangle excluding diagal
* hemm_coo_n_lo_col_conj    Conjugate transpose hermitian column major matrics defined in coo, calculate lower triangle including diagal
* hemm_coo_u_lo_col_conj    Conjugate transpose hermitian column major matrics defined in coo, calculate lower triangle excluding diagal 
* hemm_coo_n_hi_col_conj    Conjugate transpose hermitian column major matrics defined in coo, calculate higher triangle excluding diagal
* hemm_coo_u_hi_col_conj    Conjugate transpose hermitian column major matrics defined in coo, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*hermm_coo_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                     const OPENSPBLAS_SPMAT_COO *mat,
                                                     const OPENSPBLAS_Number *x,
                                                     const OPENSPBLAS_INT columns,
                                                     const OPENSPBLAS_INT ldx,
                                                     const OPENSPBLAS_Number beta,
                                                     OPENSPBLAS_Number *y,
                                                     const OPENSPBLAS_INT ldy) = {
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
    hermm_coo_n_lo_row,
    hermm_coo_u_lo_row,
    hermm_coo_n_hi_row,
    hermm_coo_u_hi_row,
    hermm_coo_n_lo_col,
    hermm_coo_u_lo_col,
    hermm_coo_n_hi_col,
    hermm_coo_u_hi_col,
    
    hermm_coo_n_lo_row_trans,
    hermm_coo_u_lo_row_trans,
    hermm_coo_n_hi_row_trans,
    hermm_coo_u_hi_row_trans,
    hermm_coo_n_lo_col_trans,
    hermm_coo_u_lo_col_trans,
    hermm_coo_n_hi_col_trans,
    hermm_coo_u_hi_col_trans,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmm_coo_n_lo_row         triangular row major matrics defined in coo, calculate lower triangle including diagal
* trmm_coo_u_lo_row         triangular row major matrics defined in coo, calculate lower triangle excluding diagal
* trmm_coo_n_hi_row         triangular row major matrics defined in coo, calculate higher triangle excluding diagal
* trmm_coo_u_hi_row         triangular row major matrics defined in coo, calculate higher triangle excluding diagal
* trmm_coo_n_lo_col         triangular column major matrics defined in coo, calculate lower triangle including diagal
* trmm_coo_u_lo_col         triangular column major matrics defined in coo, calculate lower triangle excluding diagal 
* trmm_coo_n_hi_col         triangular column major matrics defined in coo, calculate higher triangle excluding diagal
* trmm_coo_u_hi_col         triangular column major matrics defined in coo, calculate higher triangle excluding diagal
* trmm_coo_n_lo_row_conj    Conjugate transpose triangular row major matrics defined in coo, calculate lower triangle including diagal
* trmm_coo_u_lo_row_conj    Conjugate transpose triangular row major matrics defined in coo, calculate lower triangle excluding diagal
* trmm_coo_n_hi_row_conj    Conjugate transpose triangular row major matrics defined in coo, calculate higher triangle excluding diagal
* trmm_coo_u_hi_row_conj    Conjugate transpose triangular row major matrics defined in coo, calculate higher triangle excluding diagal
* trmm_coo_n_lo_col_conj    Conjugate transpose triangular column major matrics defined in coo, calculate lower triangle including diagal
* trmm_coo_u_lo_col_conj    Conjugate transpose triangular column major matrics defined in coo, calculate lower triangle excluding diagal 
* trmm_coo_n_hi_col_conj    Conjugate transpose triangular column major matrics defined in coo, calculate higher triangle excluding diagal
* trmm_coo_u_hi_col_conj    Conjugate transpose triangular column major matrics defined in coo, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*trmm_coo_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                                   const OPENSPBLAS_SPMAT_COO *mat,
                                                                   const OPENSPBLAS_Number *x,
                                                                   const OPENSPBLAS_INT columns,
                                                                   const OPENSPBLAS_INT ldx,
                                                                   const OPENSPBLAS_Number beta,
                                                                   OPENSPBLAS_Number *y,
                                                                   const OPENSPBLAS_INT ldy) = {
    trmm_coo_n_lo_row,
    trmm_coo_u_lo_row,
    trmm_coo_n_hi_row,
    trmm_coo_u_hi_row,
    trmm_coo_n_lo_col,
    trmm_coo_u_lo_col,
    trmm_coo_n_hi_col,
    trmm_coo_u_hi_col,

    trmm_coo_n_lo_row_trans,
    trmm_coo_u_lo_row_trans,
    trmm_coo_n_hi_row_trans,
    trmm_coo_u_hi_row_trans,
    trmm_coo_n_lo_col_trans,
    trmm_coo_u_lo_col_trans,
    trmm_coo_n_hi_col_trans,
    trmm_coo_u_hi_col_trans,

#ifdef COMPLEX
    trmm_coo_n_lo_row_conj,
    trmm_coo_u_lo_row_conj,
    trmm_coo_n_hi_row_conj,
    trmm_coo_u_hi_row_conj,
    trmm_coo_n_lo_col_conj,
    trmm_coo_u_lo_col_conj,
    trmm_coo_n_hi_col_conj,
    trmm_coo_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmm_coo_n_row          diagonal row major matrics defined in coo, including diagal
* diagmm_coo_u_row          diagonal row major matrics defined in coo, excluding diagal
* diagmm_coo_n_col          diagonal column major matrics defined in coo, including diagal
* diagmm_coo_u_col          diagonal column major matrics defined in coo, excluding diagal
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

static openspblas_sparse_status_t (*diagmm_coo_diag_layout[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_COO *mat,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_INT columns,
                                                      const OPENSPBLAS_INT ldx,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y,
                                                      const OPENSPBLAS_INT ldy) = {
    diagmm_coo_n_row,
    diagmm_coo_u_row,
    diagmm_coo_n_col,
    diagmm_coo_u_col,
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* gemm_csc_row          General row major matrics defined in csc storage format
* gemm_csc_col          General column major matrics defined in csc storage format 
* gemm_csc_row_trans    Transpose of general column major matrics defined in csc storage format
* gemm_csc_col_trans    Transpose of general column major matrics defined in csc storage format
* gemm_csc_row_conj     Conjugate transpose of general column major matrics defined in csc storage format
* gemm_csc_col_conj     Conjugate transpose of general column major matrics defined in csc storage format 
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



static openspblas_sparse_status_t (*gemm_csc_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_CSC *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    gemm_csc_row,
    gemm_csc_col,
    gemm_csc_row_trans,
    gemm_csc_col_trans,
#ifdef COMPLEX
    gemm_csc_row_conj,
    gemm_csc_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* symm_csr_n_lo_row         symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row         symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col         symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col         symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*symm_csc_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_CSC *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    symm_csc_n_lo_row,
    symm_csc_u_lo_row,
    symm_csc_n_hi_row,
    symm_csc_u_hi_row,
    symm_csc_n_lo_col,
    symm_csc_u_lo_col,
    symm_csc_n_hi_col,
    symm_csc_u_hi_col,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_csc_n_lo_row_conj,
    symm_csc_u_lo_row_conj,
    symm_csc_n_hi_row_conj,
    symm_csc_u_hi_row_conj,
    symm_csc_n_lo_col_conj,
    symm_csc_u_lo_col_conj,
    symm_csc_n_hi_col_conj,
    symm_csc_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* hemm_csc_n_lo_row         hermitian row major matrics defined in csc, calculate lower triangle including diagal
* hemm_csc_u_lo_row         hermitian row major matrics defined in csc, calculate lower triangle excluding diagal
* hemm_csc_n_hi_row         hermitian row major matrics defined in csc, calculate higher triangle excluding diagal
* hemm_csc_u_hi_row         hermitian row major matrics defined in csc, calculate higher triangle excluding diagal
* hemm_csc_n_lo_col         hermitian column major matrics defined in csc, calculate lower triangle including diagal
* hemm_csc_u_lo_col         hermitian column major matrics defined in csc, calculate lower triangle excluding diagal 
* hemm_csc_n_hi_col         hermitian column major matrics defined in csc, calculate higher triangle excluding diagal
* hemm_csc_u_hi_col         hermitian column major matrics defined in csc, calculate higher triangle excluding diagal
* hemm_csc_n_lo_row_conj    Conjugate transpose hermitian row major matrics defined in csc, calculate lower triangle including diagal
* hemm_csc_u_lo_row_conj    Conjugate transpose hermitian row major matrics defined in csc, calculate lower triangle excluding diagal
* hemm_csc_n_hi_row_conj    Conjugate transpose hermitian row major matrics defined in csc, calculate higher triangle excluding diagal
* hemm_csc_u_hi_row_conj    Conjugate transpose hermitian row major matrics defined in csc, calculate higher triangle excluding diagal
* hemm_csc_n_lo_col_conj    Conjugate transpose hermitian column major matrics defined in csc, calculate lower triangle including diagal
* hemm_csc_u_lo_col_conj    Conjugate transpose hermitian column major matrics defined in csc, calculate lower triangle excluding diagal 
* hemm_csc_n_hi_col_conj    Conjugate transpose hermitian column major matrics defined in csc, calculate higher triangle excluding diagal
* hemm_csc_u_hi_col_conj    Conjugate transpose hermitian column major matrics defined in csc, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*hermm_csc_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                     const OPENSPBLAS_SPMAT_CSC *mat,
                                                     const OPENSPBLAS_Number *x,
                                                     const OPENSPBLAS_INT columns,
                                                     const OPENSPBLAS_INT ldx,
                                                     const OPENSPBLAS_Number beta,
                                                     OPENSPBLAS_Number *y,
                                                     const OPENSPBLAS_INT ldy) = {
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
    hermm_csc_n_lo_row,
    hermm_csc_u_lo_row,
    hermm_csc_n_hi_row,
    hermm_csc_u_hi_row,
    hermm_csc_n_lo_col,
    hermm_csc_u_lo_col,
    hermm_csc_n_hi_col,
    hermm_csc_u_hi_col,
    
    hermm_csc_n_lo_row_trans,
    hermm_csc_u_lo_row_trans,
    hermm_csc_n_hi_row_trans,
    hermm_csc_u_hi_row_trans,
    hermm_csc_n_lo_col_trans,
    hermm_csc_u_lo_col_trans,
    hermm_csc_n_hi_col_trans,
    hermm_csc_u_hi_col_trans,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmm_csc_n_lo_row         triangular row major matrics defined in csc, calculate lower triangle including diagal
* trmm_csc_u_lo_row         triangular row major matrics defined in csc, calculate lower triangle excluding diagal
* trmm_csc_n_hi_row         triangular row major matrics defined in csc, calculate higher triangle excluding diagal
* trmm_csc_u_hi_row         triangular row major matrics defined in csc, calculate higher triangle excluding diagal
* trmm_csc_n_lo_col         triangular column major matrics defined in csc, calculate lower triangle including diagal
* trmm_csc_u_lo_col         triangular column major matrics defined in csc, calculate lower triangle excluding diagal 
* trmm_csc_n_hi_col         triangular column major matrics defined in csc, calculate higher triangle excluding diagal
* trmm_csc_u_hi_col         triangular column major matrics defined in csc, calculate higher triangle excluding diagal
* trmm_csc_n_lo_row_conj    Conjugate transpose triangular row major matrics defined in csc, calculate lower triangle including diagal
* trmm_csc_u_lo_row_conj    Conjugate transpose triangular row major matrics defined in csc, calculate lower triangle excluding diagal
* trmm_csc_n_hi_row_conj    Conjugate transpose triangular row major matrics defined in csc, calculate higher triangle excluding diagal
* trmm_csc_u_hi_row_conj    Conjugate transpose triangular row major matrics defined in csc, calculate higher triangle excluding diagal
* trmm_csc_n_lo_col_conj    Conjugate transpose triangular column major matrics defined in csc, calculate lower triangle including diagal
* trmm_csc_u_lo_col_conj    Conjugate transpose triangular column major matrics defined in csc, calculate lower triangle excluding diagal 
* trmm_csc_n_hi_col_conj    Conjugate transpose triangular column major matrics defined in csc, calculate higher triangle excluding diagal
* trmm_csc_u_hi_col_conj    Conjugate transpose triangular column major matrics defined in csc, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*trmm_csc_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                                   const OPENSPBLAS_SPMAT_CSC *mat,
                                                                   const OPENSPBLAS_Number *x,
                                                                   const OPENSPBLAS_INT columns,
                                                                   const OPENSPBLAS_INT ldx,
                                                                   const OPENSPBLAS_Number beta,
                                                                   OPENSPBLAS_Number *y,
                                                                   const OPENSPBLAS_INT ldy) = {
    trmm_csc_n_lo_row,
    trmm_csc_u_lo_row,
    trmm_csc_n_hi_row,
    trmm_csc_u_hi_row,
    trmm_csc_n_lo_col,
    trmm_csc_u_lo_col,
    trmm_csc_n_hi_col,
    trmm_csc_u_hi_col,

    trmm_csc_n_lo_row_trans,
    trmm_csc_u_lo_row_trans,
    trmm_csc_n_hi_row_trans,
    trmm_csc_u_hi_row_trans,
    trmm_csc_n_lo_col_trans,
    trmm_csc_u_lo_col_trans,
    trmm_csc_n_hi_col_trans,
    trmm_csc_u_hi_col_trans,
#ifdef COMPLEX
    trmm_csc_n_lo_row_conj,
    trmm_csc_u_lo_row_conj,
    trmm_csc_n_hi_row_conj,
    trmm_csc_u_hi_row_conj,
    trmm_csc_n_lo_col_conj,
    trmm_csc_u_lo_col_conj,
    trmm_csc_n_hi_col_conj,
    trmm_csc_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmm_csc_n_row          diagonal row major matrics defined in csc, including diagal
* diagmm_csc_u_row          diagonal row major matrics defined in csc, excluding diagal
* diagmm_csc_n_col          diagonal column major matrics defined in csc, including diagal
* diagmm_csc_u_col          diagonal column major matrics defined in csc, excluding diagal
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

static openspblas_sparse_status_t (*diagmm_csc_diag_layout[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_CSC *mat,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_INT columns,
                                                      const OPENSPBLAS_INT ldx,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y,
                                                      const OPENSPBLAS_INT ldy) = {
    diagmm_csc_n_row,
    diagmm_csc_u_row,
    diagmm_csc_n_col,
    diagmm_csc_u_col,
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* gemm_bsr_row          General row major matrics defined in bsr storage format
* gemm_bsr_col          General column major matrics defined in bsr storage format 
* gemm_bsr_row_trans    Transpose of general column major matrics defined in bsr storage format
* gemm_bsr_col_trans    Transpose of general column major matrics defined in bsr storage format
* gemm_bsr_row_conj     Conjugate transpose of general column major matrics defined in bsr storage format
* gemm_bsr_col_conj     Conjugate transpose of general column major matrics defined in bsr storage format 
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

static openspblas_sparse_status_t (*gemm_bsr_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_BSR *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    gemm_bsr_row,
    gemm_bsr_col,
    gemm_bsr_row_trans,
    gemm_bsr_col_trans,
#ifdef COMPLEX
    gemm_bsr_row_conj,
    gemm_bsr_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* symm_csr_n_lo_row         symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row         symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col         symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col         symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*symm_bsr_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_BSR *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    symm_bsr_n_lo_row,
    symm_bsr_u_lo_row,
    symm_bsr_n_hi_row,
    symm_bsr_u_hi_row,
    symm_bsr_n_lo_col,
    symm_bsr_u_lo_col,
    symm_bsr_n_hi_col,
    symm_bsr_u_hi_col,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_bsr_n_lo_row_conj,
    symm_bsr_u_lo_row_conj,
    symm_bsr_n_hi_row_conj,
    symm_bsr_u_hi_row_conj,
    symm_bsr_n_lo_col_conj,
    symm_bsr_u_lo_col_conj,
    symm_bsr_n_hi_col_conj,
    symm_bsr_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* hemm_bsr_n_lo_row         hermitian row major matrics defined in bsr, calculate lower triangle including diagal
* hemm_bsr_u_lo_row         hermitian row major matrics defined in bsr, calculate lower triangle excluding diagal
* hemm_bsr_n_hi_row         hermitian row major matrics defined in bsr, calculate higher triangle excluding diagal
* hemm_bsr_u_hi_row         hermitian row major matrics defined in bsr, calculate higher triangle excluding diagal
* hemm_bsr_n_lo_col         hermitian column major matrics defined in bsr, calculate lower triangle including diagal
* hemm_bsr_u_lo_col         hermitian column major matrics defined in bsr, calculate lower triangle excluding diagal 
* hemm_bsr_n_hi_col         hermitian column major matrics defined in bsr, calculate higher triangle excluding diagal
* hemm_bsr_u_hi_col         hermitian column major matrics defined in bsr, calculate higher triangle excluding diagal
* hemm_bsr_n_lo_row_conj    Conjugate transpose hermitian row major matrics defined in bsr, calculate lower triangle including diagal
* hemm_bsr_u_lo_row_conj    Conjugate transpose hermitian row major matrics defined in bsr, calculate lower triangle excluding diagal
* hemm_bsr_n_hi_row_conj    Conjugate transpose hermitian row major matrics defined in bsr, calculate higher triangle excluding diagal
* hemm_bsr_u_hi_row_conj    Conjugate transpose hermitian row major matrics defined in bsr, calculate higher triangle excluding diagal
* hemm_bsr_n_lo_col_conj    Conjugate transpose hermitian column major matrics defined in bsr, calculate lower triangle including diagal
* hemm_bsr_u_lo_col_conj    Conjugate transpose hermitian column major matrics defined in bsr, calculate lower triangle excluding diagal 
* hemm_bsr_n_hi_col_conj    Conjugate transpose hermitian column major matrics defined in bsr, calculate higher triangle excluding diagal
* hemm_bsr_u_hi_col_conj    Conjugate transpose hermitian column major matrics defined in bsr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*hermm_bsr_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                     const OPENSPBLAS_SPMAT_BSR *mat,
                                                     const OPENSPBLAS_Number *x,
                                                     const OPENSPBLAS_INT columns,
                                                     const OPENSPBLAS_INT ldx,
                                                     const OPENSPBLAS_Number beta,
                                                     OPENSPBLAS_Number *y,
                                                     const OPENSPBLAS_INT ldy) = {
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
    hermm_bsr_n_lo_row,
    hermm_bsr_u_lo_row,
    hermm_bsr_n_hi_row,
    hermm_bsr_u_hi_row,
    hermm_bsr_n_lo_col,
    hermm_bsr_u_lo_col,
    hermm_bsr_n_hi_col,
    hermm_bsr_u_hi_col,
    
    hermm_bsr_n_lo_row_trans,
    hermm_bsr_u_lo_row_trans,
    hermm_bsr_n_hi_row_trans,
    hermm_bsr_u_hi_row_trans,
    hermm_bsr_n_lo_col_trans,
    hermm_bsr_u_lo_col_trans,
    hermm_bsr_n_hi_col_trans,
    hermm_bsr_u_hi_col_trans,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmm_bsr_n_lo_row         triangular row major matrics defined in bsr, calculate lower triangle including diagal
* trmm_bsr_u_lo_row         triangular row major matrics defined in bsr, calculate lower triangle excluding diagal
* trmm_bsr_n_hi_row         triangular row major matrics defined in bsr, calculate higher triangle excluding diagal
* trmm_bsr_u_hi_row         triangular row major matrics defined in bsr, calculate higher triangle excluding diagal
* trmm_bsr_n_lo_col         triangular column major matrics defined in bsr, calculate lower triangle including diagal
* trmm_bsr_u_lo_col         triangular column major matrics defined in bsr, calculate lower triangle excluding diagal 
* trmm_bsr_n_hi_col         triangular column major matrics defined in bsr, calculate higher triangle excluding diagal
* trmm_bsr_u_hi_col         triangular column major matrics defined in bsr, calculate higher triangle excluding diagal
* trmm_bsr_n_lo_row_conj    Conjugate transpose triangular row major matrics defined in bsr, calculate lower triangle including diagal
* trmm_bsr_u_lo_row_conj    Conjugate transpose triangular row major matrics defined in bsr, calculate lower triangle excluding diagal
* trmm_bsr_n_hi_row_conj    Conjugate transpose triangular row major matrics defined in bsr, calculate higher triangle excluding diagal
* trmm_bsr_u_hi_row_conj    Conjugate transpose triangular row major matrics defined in bsr, calculate higher triangle excluding diagal
* trmm_bsr_n_lo_col_conj    Conjugate transpose triangular column major matrics defined in bsr, calculate lower triangle including diagal
* trmm_bsr_u_lo_col_conj    Conjugate transpose triangular column major matrics defined in bsr, calculate lower triangle excluding diagal 
* trmm_bsr_n_hi_col_conj    Conjugate transpose triangular column major matrics defined in bsr, calculate higher triangle excluding diagal
* trmm_bsr_u_hi_col_conj    Conjugate transpose triangular column major matrics defined in bsr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*trmm_bsr_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                                   const OPENSPBLAS_SPMAT_BSR *mat,
                                                                   const OPENSPBLAS_Number *x,
                                                                   const OPENSPBLAS_INT columns,
                                                                   const OPENSPBLAS_INT ldx,
                                                                   const OPENSPBLAS_Number beta,
                                                                   OPENSPBLAS_Number *y,
                                                                   const OPENSPBLAS_INT ldy) = {
    trmm_bsr_n_lo_row,
    trmm_bsr_u_lo_row,
    trmm_bsr_n_hi_row,
    trmm_bsr_u_hi_row,
    trmm_bsr_n_lo_col,
    trmm_bsr_u_lo_col,
    trmm_bsr_n_hi_col,
    trmm_bsr_u_hi_col,
    trmm_bsr_n_lo_row_trans,
    trmm_bsr_u_lo_row_trans,
    trmm_bsr_n_hi_row_trans,
    trmm_bsr_u_hi_row_trans,
    trmm_bsr_n_lo_col_trans,
    trmm_bsr_u_lo_col_trans,
    trmm_bsr_n_hi_col_trans,
    trmm_bsr_u_hi_col_trans,
#ifdef COMPLEX
    trmm_bsr_n_lo_row_conj,
    trmm_bsr_u_lo_row_conj,
    trmm_bsr_n_hi_row_conj,
    trmm_bsr_u_hi_row_conj,
    trmm_bsr_n_lo_col_conj,
    trmm_bsr_u_lo_col_conj,
    trmm_bsr_n_hi_col_conj,
    trmm_bsr_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmm_bsr_n_row          diagonal row major matrics defined in bsr, including diagal
* diagmm_bsr_u_row          diagonal row major matrics defined in bsr, excluding diagal
* diagmm_bsr_n_col          diagonal column major matrics defined in bsr, including diagal
* diagmm_bsr_u_col          diagonal column major matrics defined in bsr, excluding diagal
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

static openspblas_sparse_status_t (*diagmm_bsr_diag_layout[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_BSR *mat,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_INT columns,
                                                      const OPENSPBLAS_INT ldx,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y,
                                                      const OPENSPBLAS_INT ldy) = {
    diagmm_bsr_n_row,
    diagmm_bsr_u_row,
    diagmm_bsr_n_col,
    diagmm_bsr_u_col,
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* gemm_bsr_row          General row major matrics defined in bsr storage format
* gemm_bsr_col          General column major matrics defined in bsr storage format 
* gemm_bsr_row_trans    Transpose of general column major matrics defined in bsr storage format
* gemm_bsr_col_trans    Transpose of general column major matrics defined in bsr storage format
* gemm_bsr_row_conj     Conjugate transpose of general column major matrics defined in bsr storage format
* gemm_bsr_col_conj     Conjugate transpose of general column major matrics defined in bsr storage format 
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

static openspblas_sparse_status_t (*gemm_sky_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_SKY *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    gemm_sky_row,
    gemm_sky_col,
    gemm_sky_row_trans,
    gemm_sky_col_trans,
#ifdef COMPLEX
    NULL,
    NULL,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* symm_csr_n_lo_row         symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row         symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col         symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col         symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*symm_sky_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_SKY *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    symm_sky_n_lo_row,
    symm_sky_u_lo_row,
    symm_sky_n_hi_row,
    symm_sky_u_hi_row,
    symm_sky_n_lo_col,
    symm_sky_u_lo_col,
    symm_sky_n_hi_col,
    symm_sky_u_hi_col,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_sky_n_lo_row_conj,
    symm_sky_u_lo_row_conj,
    symm_sky_n_hi_row_conj,
    symm_sky_u_hi_row_conj,
    symm_sky_n_lo_col_conj,
    symm_sky_u_lo_col_conj,
    symm_sky_n_hi_col_conj,
    symm_sky_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* hemm_sky_n_lo_row         hermitian row major matrics defined in sky, calculate lower triangle including diagal
* hemm_sky_u_lo_row         hermitian row major matrics defined in sky, calculate lower triangle excluding diagal
* hemm_sky_n_hi_row         hermitian row major matrics defined in sky, calculate higher triangle excluding diagal
* hemm_sky_u_hi_row         hermitian row major matrics defined in sky, calculate higher triangle excluding diagal
* hemm_sky_n_lo_col         hermitian column major matrics defined in sky, calculate lower triangle including diagal
* hemm_sky_u_lo_col         hermitian column major matrics defined in sky, calculate lower triangle excluding diagal 
* hemm_sky_n_hi_col         hermitian column major matrics defined in sky, calculate higher triangle excluding diagal
* hemm_sky_u_hi_col         hermitian column major matrics defined in sky, calculate higher triangle excluding diagal
* hemm_sky_n_lo_row_conj    Conjugate transpose hermitian row major matrics defined in sky, calculate lower triangle including diagal
* hemm_sky_u_lo_row_conj    Conjugate transpose hermitian row major matrics defined in sky, calculate lower triangle excluding diagal
* hemm_sky_n_hi_row_conj    Conjugate transpose hermitian row major matrics defined in sky, calculate higher triangle excluding diagal
* hemm_sky_u_hi_row_conj    Conjugate transpose hermitian row major matrics defined in sky, calculate higher triangle excluding diagal
* hemm_sky_n_lo_col_conj    Conjugate transpose hermitian column major matrics defined in sky, calculate lower triangle including diagal
* hemm_sky_u_lo_col_conj    Conjugate transpose hermitian column major matrics defined in sky, calculate lower triangle excluding diagal 
* hemm_sky_n_hi_col_conj    Conjugate transpose hermitian column major matrics defined in sky, calculate higher triangle excluding diagal
* hemm_sky_u_hi_col_conj    Conjugate transpose hermitian column major matrics defined in sky, calculate higher triangle excluding diagal

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
static openspblas_sparse_status_t (*hermm_sky_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                     const OPENSPBLAS_SPMAT_SKY *mat,
                                                     const OPENSPBLAS_Number *x,
                                                     const OPENSPBLAS_INT columns,
                                                     const OPENSPBLAS_INT ldx,
                                                     const OPENSPBLAS_Number beta,
                                                     OPENSPBLAS_Number *y,
                                                     const OPENSPBLAS_INT ldy) = {
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
    hermm_sky_n_lo_row,
    hermm_sky_u_lo_row,
    hermm_sky_n_hi_row,
    hermm_sky_u_hi_row,
    hermm_sky_n_lo_col,
    hermm_sky_u_lo_col,
    hermm_sky_n_hi_col,
    hermm_sky_u_hi_col,
    
    hermm_sky_n_lo_row_trans,
    hermm_sky_u_lo_row_trans,
    hermm_sky_n_hi_row_trans,
    hermm_sky_u_hi_row_trans,
    hermm_sky_n_lo_col_trans,
    hermm_sky_u_lo_col_trans,
    hermm_sky_n_hi_col_trans,
    hermm_sky_u_hi_col_trans,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmm_sky_n_lo_row         triangular row major matrics defined in sky, calculate lower triangle including diagal
* trmm_sky_u_lo_row         triangular row major matrics defined in sky, calculate lower triangle excluding diagal
* trmm_sky_n_hi_row         triangular row major matrics defined in sky, calculate higher triangle excluding diagal
* trmm_sky_u_hi_row         triangular row major matrics defined in sky, calculate higher triangle excluding diagal
* trmm_sky_n_lo_col         triangular column major matrics defined in sky, calculate lower triangle including diagal
* trmm_sky_u_lo_col         triangular column major matrics defined in sky, calculate lower triangle excluding diagal 
* trmm_sky_n_hi_col         triangular column major matrics defined in sky, calculate higher triangle excluding diagal
* trmm_sky_u_hi_col         triangular column major matrics defined in sky, calculate higher triangle excluding diagal
* trmm_sky_n_lo_row_conj    Conjugate transpose triangular row major matrics defined in sky, calculate lower triangle including diagal
* trmm_sky_u_lo_row_conj    Conjugate transpose triangular row major matrics defined in sky, calculate lower triangle excluding diagal
* trmm_sky_n_hi_row_conj    Conjugate transpose triangular row major matrics defined in sky, calculate higher triangle excluding diagal
* trmm_sky_u_hi_row_conj    Conjugate transpose triangular row major matrics defined in sky, calculate higher triangle excluding diagal
* trmm_sky_n_lo_col_conj    Conjugate transpose triangular column major matrics defined in sky, calculate lower triangle including diagal
* trmm_sky_u_lo_col_conj    Conjugate transpose triangular column major matrics defined in sky, calculate lower triangle excluding diagal 
* trmm_sky_n_hi_col_conj    Conjugate transpose triangular column major matrics defined in sky, calculate higher triangle excluding diagal
* trmm_sky_u_hi_col_conj    Conjugate transpose triangular column major matrics defined in sky, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*trmm_sky_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                                   const OPENSPBLAS_SPMAT_SKY *mat,
                                                                   const OPENSPBLAS_Number *x,
                                                                   const OPENSPBLAS_INT columns,
                                                                   const OPENSPBLAS_INT ldx,
                                                                   const OPENSPBLAS_Number beta,
                                                                   OPENSPBLAS_Number *y,
                                                                   const OPENSPBLAS_INT ldy) = {
    trmm_sky_n_lo_row,
    trmm_sky_u_lo_row,
    trmm_sky_n_hi_row,
    trmm_sky_u_hi_row,
    trmm_sky_n_lo_col,
    trmm_sky_u_lo_col,
    trmm_sky_n_hi_col,
    trmm_sky_u_hi_col,
    trmm_sky_n_lo_row_trans,
    trmm_sky_u_lo_row_trans,
    trmm_sky_n_hi_row_trans,
    trmm_sky_u_hi_row_trans,
    trmm_sky_n_lo_col_trans,
    trmm_sky_u_lo_col_trans,
    trmm_sky_n_hi_col_trans,
    trmm_sky_u_hi_col_trans,
#ifdef COMPLEX
    trmm_sky_n_lo_row_conj,
    trmm_sky_u_lo_row_conj,
    trmm_sky_n_hi_row_conj,
    trmm_sky_u_hi_row_conj,
    trmm_sky_n_lo_col_conj,
    trmm_sky_u_lo_col_conj,
    trmm_sky_n_hi_col_conj,
    trmm_sky_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmm_sky_n_row          diagonal row major matrics defined in sky, including diagal
* diagmm_sky_u_row          diagonal row major matrics defined in sky, excluding diagal
* diagmm_sky_n_col          diagonal column major matrics defined in sky, including diagal
* diagmm_sky_u_col          diagonal column major matrics defined in sky, excluding diagal
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

static openspblas_sparse_status_t (*diagmm_sky_diag_layout[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_SKY *mat,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_INT columns,
                                                      const OPENSPBLAS_INT ldx,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y,
                                                      const OPENSPBLAS_INT ldy) = {
    diagmm_sky_n_row,
    diagmm_sky_u_row,
    diagmm_sky_n_col,
    diagmm_sky_u_col,
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* gemm_bsr_row          General row major matrics defined in bsr storage format
* gemm_bsr_col          General column major matrics defined in bsr storage format 
* gemm_bsr_row_trans    Transpose of general column major matrics defined in bsr storage format
* gemm_bsr_col_trans    Transpose of general column major matrics defined in bsr storage format
* gemm_bsr_row_conj     Conjugate transpose of general column major matrics defined in bsr storage format
* gemm_bsr_col_conj     Conjugate transpose of general column major matrics defined in bsr storage format 
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

static openspblas_sparse_status_t (*gemm_dia_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_DIA *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    gemm_dia_row,
    gemm_dia_col,
    gemm_dia_row_trans,
    gemm_dia_col_trans,
#ifdef COMPLEX
    gemm_dia_row_conj,
    gemm_dia_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* symm_csr_n_lo_row         symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row         symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row         symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col         symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col         symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col         symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate lower triangle excluding diagal
* symm_csr_n_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_row_conj    Conjugate transpose symmetric row major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_n_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle including diagal
* symm_csr_u_lo_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate lower triangle excluding diagal 
* symm_csr_n_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal
* symm_csr_u_hi_col_conj    Conjugate transpose symmetric column major matrics defined in csr, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*symm_dia_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                         const OPENSPBLAS_SPMAT_DIA *mat,
                                                         const OPENSPBLAS_Number *x,
                                                         const OPENSPBLAS_INT columns,
                                                         const OPENSPBLAS_INT ldx,
                                                         const OPENSPBLAS_Number beta,
                                                         OPENSPBLAS_Number *y,
                                                         const OPENSPBLAS_INT ldy) = {
    symm_dia_n_lo_row,
    symm_dia_u_lo_row,
    symm_dia_n_hi_row,
    symm_dia_u_hi_row,
    symm_dia_n_lo_col,
    symm_dia_u_lo_col,
    symm_dia_n_hi_col,
    symm_dia_u_hi_col,
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
    NULL, //padding
#ifdef COMPLEX
    symm_dia_n_lo_row_conj,
    symm_dia_u_lo_row_conj,
    symm_dia_n_hi_row_conj,
    symm_dia_u_hi_row_conj,
    symm_dia_n_lo_col_conj,
    symm_dia_u_lo_col_conj,
    symm_dia_n_hi_col_conj,
    symm_dia_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* hemm_dia_n_lo_row         hermitian row major matrics defined in dia, calculate lower triangle including diagal
* hemm_dia_u_lo_row         hermitian row major matrics defined in dia, calculate lower triangle excluding diagal
* hemm_dia_n_hi_row         hermitian row major matrics defined in dia, calculate higher triangle excluding diagal
* hemm_dia_u_hi_row         hermitian row major matrics defined in dia, calculate higher triangle excluding diagal
* hemm_dia_n_lo_col         hermitian column major matrics defined in dia, calculate lower triangle including diagal
* hemm_dia_u_lo_col         hermitian column major matrics defined in dia, calculate lower triangle excluding diagal 
* hemm_dia_n_hi_col         hermitian column major matrics defined in dia, calculate higher triangle excluding diagal
* hemm_dia_u_hi_col         hermitian column major matrics defined in dia, calculate higher triangle excluding diagal
* hemm_dia_n_lo_row_conj    Conjugate transpose hermitian row major matrics defined in dia, calculate lower triangle including diagal
* hemm_dia_u_lo_row_conj    Conjugate transpose hermitian row major matrics defined in dia, calculate lower triangle excluding diagal
* hemm_dia_n_hi_row_conj    Conjugate transpose hermitian row major matrics defined in dia, calculate higher triangle excluding diagal
* hemm_dia_u_hi_row_conj    Conjugate transpose hermitian row major matrics defined in dia, calculate higher triangle excluding diagal
* hemm_dia_n_lo_col_conj    Conjugate transpose hermitian column major matrics defined in dia, calculate lower triangle including diagal
* hemm_dia_u_lo_col_conj    Conjugate transpose hermitian column major matrics defined in dia, calculate lower triangle excluding diagal 
* hemm_dia_n_hi_col_conj    Conjugate transpose hermitian column major matrics defined in dia, calculate higher triangle excluding diagal
* hemm_dia_u_hi_col_conj    Conjugate transpose hermitian column major matrics defined in dia, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*hermm_dia_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                     const OPENSPBLAS_SPMAT_DIA *mat,
                                                     const OPENSPBLAS_Number *x,
                                                     const OPENSPBLAS_INT columns,
                                                     const OPENSPBLAS_INT ldx,
                                                     const OPENSPBLAS_Number beta,
                                                     OPENSPBLAS_Number *y,
                                                     const OPENSPBLAS_INT ldy) = {
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
    hermm_dia_n_lo_row,
    hermm_dia_u_lo_row,
    hermm_dia_n_hi_row,
    hermm_dia_u_hi_row,
    hermm_dia_n_lo_col,
    hermm_dia_u_lo_col,
    hermm_dia_n_hi_col,
    hermm_dia_u_hi_col,
    
    hermm_dia_n_lo_row_trans,
    hermm_dia_u_lo_row_trans,
    hermm_dia_n_hi_row_trans,
    hermm_dia_u_hi_row_trans,
    hermm_dia_n_lo_col_trans,
    hermm_dia_u_lo_col_trans,
    hermm_dia_n_hi_col_trans,
    hermm_dia_u_hi_col_trans,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* trmm_dia_n_lo_row         triangular row major matrics defined in dia, calculate lower triangle including diagal
* trmm_dia_u_lo_row         triangular row major matrics defined in dia, calculate lower triangle excluding diagal
* trmm_dia_n_hi_row         triangular row major matrics defined in dia, calculate higher triangle excluding diagal
* trmm_dia_u_hi_row         triangular row major matrics defined in dia, calculate higher triangle excluding diagal
* trmm_dia_n_lo_col         triangular column major matrics defined in dia, calculate lower triangle including diagal
* trmm_dia_u_lo_col         triangular column major matrics defined in dia, calculate lower triangle excluding diagal 
* trmm_dia_n_hi_col         triangular column major matrics defined in dia, calculate higher triangle excluding diagal
* trmm_dia_u_hi_col         triangular column major matrics defined in dia, calculate higher triangle excluding diagal
* trmm_dia_n_lo_row_conj    Conjugate transpose triangular row major matrics defined in dia, calculate lower triangle including diagal
* trmm_dia_u_lo_row_conj    Conjugate transpose triangular row major matrics defined in dia, calculate lower triangle excluding diagal
* trmm_dia_n_hi_row_conj    Conjugate transpose triangular row major matrics defined in dia, calculate higher triangle excluding diagal
* trmm_dia_u_hi_row_conj    Conjugate transpose triangular row major matrics defined in dia, calculate higher triangle excluding diagal
* trmm_dia_n_lo_col_conj    Conjugate transpose triangular column major matrics defined in dia, calculate lower triangle including diagal
* trmm_dia_u_lo_col_conj    Conjugate transpose triangular column major matrics defined in dia, calculate lower triangle excluding diagal 
* trmm_dia_n_hi_col_conj    Conjugate transpose triangular column major matrics defined in dia, calculate higher triangle excluding diagal
* trmm_dia_u_hi_col_conj    Conjugate transpose triangular column major matrics defined in dia, calculate higher triangle excluding diagal

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

static openspblas_sparse_status_t (*trmm_dia_diag_fill_layout_operation[])(const OPENSPBLAS_Number alpha,
                                                                   const OPENSPBLAS_SPMAT_DIA *mat,
                                                                   const OPENSPBLAS_Number *x,
                                                                   const OPENSPBLAS_INT columns,
                                                                   const OPENSPBLAS_INT ldx,
                                                                   const OPENSPBLAS_Number beta,
                                                                   OPENSPBLAS_Number *y,
                                                                   const OPENSPBLAS_INT ldy) = {
    trmm_dia_n_lo_row,
    trmm_dia_u_lo_row,
    trmm_dia_n_hi_row,
    trmm_dia_u_hi_row,
    trmm_dia_n_lo_col,
    trmm_dia_u_lo_col,
    trmm_dia_n_hi_col,
    trmm_dia_u_hi_col,
    trmm_dia_n_lo_row_trans,
    trmm_dia_u_lo_row_trans,
    trmm_dia_n_hi_row_trans,
    trmm_dia_u_hi_row_trans,
    trmm_dia_n_lo_col_trans,
    trmm_dia_u_lo_col_trans,
    trmm_dia_n_hi_col_trans,
    trmm_dia_u_hi_col_trans,
#ifdef COMPLEX
    trmm_dia_n_lo_row_conj,
    trmm_dia_u_lo_row_conj,
    trmm_dia_n_hi_row_conj,
    trmm_dia_u_hi_row_conj,
    trmm_dia_n_lo_col_conj,
    trmm_dia_u_lo_col_conj,
    trmm_dia_n_hi_col_conj,
    trmm_dia_u_hi_col_conj,
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* diagmm_dia_n_row          diagonal row major matrics defined in dia, including diagal
* diagmm_dia_u_row          diagonal row major matrics defined in dia, excluding diagal
* diagmm_dia_n_col          diagonal column major matrics defined in dia, including diagal
* diagmm_dia_u_col          diagonal column major matrics defined in dia, excluding diagal
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

static openspblas_sparse_status_t (*diagmm_dia_diag_layout[])(const OPENSPBLAS_Number alpha,
                                                      const OPENSPBLAS_SPMAT_DIA *mat,
                                                      const OPENSPBLAS_Number *x,
                                                      const OPENSPBLAS_INT columns,
                                                      const OPENSPBLAS_INT ldx,
                                                      const OPENSPBLAS_Number beta,
                                                      OPENSPBLAS_Number *y,
                                                      const OPENSPBLAS_INT ldy) = {
    diagmm_dia_n_row,
    diagmm_dia_u_row,
    diagmm_dia_n_col,
    diagmm_dia_u_col,
};

openspblas_sparse_status_t ONAME(const openspblas_sparse_operation_t operation,
                          const OPENSPBLAS_Number alpha,
                          const openspblas_sparse_matrix_t A,
                          const struct openspblas_matrix_descr descr, /* openspblas_sparse_matrix_type_t + openspblas_sparse_fill_mode_t + openspblas_sparse_diag_type_t */
                          const openspblas_sparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                          const OPENSPBLAS_Number *x,
                          const OPENSPBLAS_INT columns,
                          const OPENSPBLAS_INT ldx,
                          const OPENSPBLAS_Number beta,
                          OPENSPBLAS_Number *y,
                          const OPENSPBLAS_INT ldy)
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
            check_null_return(gemm_csr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_csr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_csr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_csr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_csr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_csr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_csr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_csr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_csr_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_csr_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == OPENSPBLAS_SPARSE_FORMAT_COO)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_coo_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_coo_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_coo_diag_fill_layout[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_coo_diag_fill_layout[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_coo_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_coo_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_coo_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_coo_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_coo_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_coo_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
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
            check_null_return(gemm_csc_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_csc_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_csc_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_csc_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_csc_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_csc_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_csc_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_csc_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_csc_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_csc_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == OPENSPBLAS_SPARSE_FORMAT_BSR)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_bsr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_bsr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_bsr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_bsr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_bsr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_bsr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_bsr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_bsr_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_bsr_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_bsr_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == OPENSPBLAS_SPARSE_FORMAT_SKY)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_sky_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_sky_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_sky_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_sky_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_sky_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_sky_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_sky_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_sky_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == OPENSPBLAS_SPARSE_FORMAT_DIA)
    {
        if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_GENERAL)
        {
            check_null_return(gemm_dia_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return gemm_dia_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            check_null_return(symm_dia_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return symm_dia_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            check_null_return(hermm_dia_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return hermm_dia_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            check_null_return(trmm_dia_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return trmm_dia_diag_fill_layout_operation[index4(operation, layout, descr.mode, descr.diag, OPENSPBLAS_SPARSE_LAYOUT_NUM, OPENSPBLAS_SPARSE_FILL_MODE_NUM, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else if (descr.type == OPENSPBLAS_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            check_null_return(diagmm_dia_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
            return diagmm_dia_diag_layout[index2(layout, descr.diag, OPENSPBLAS_SPARSE_DIAG_TYPE_NUM)](alpha, A->mat, x, columns, ldx, beta, y, ldy);
        }
        else
        {
            return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else
    {
        return OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED;
    }
}
