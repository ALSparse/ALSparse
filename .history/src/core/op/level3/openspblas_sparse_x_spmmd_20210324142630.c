/**
 * @brief implement for openspblas_sparse_?_mm intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "openspblas/util.h"
#include "openspblas/opt.h"
#include "openspblas/spapi.h"
#include "openspblas/kernel.h"

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* spmmd_csr_row          General row major matrics defined in csr storage format
* spmmd_csr_col          General column major matrics defined in csr storage format 
* spmmd_csr_row_trans    Transpose of general column major matrics defined in csr storage format
* spmmd_csr_col_trans    Transpose of general column major matrics defined in csr storage format
* spmmd_csr_row_conj     Conjugate transpose of general column major matrics defined in csr storage format
* spmmd_csr_col_conj     Conjugate transpose of general column major matrics defined in csr storage format 
* C := op(A)*B
*
* A         a sparse matrix
* B         a sparse matrix
* C         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* matA      a sparse matrix 
* matB      a sparse matrix
* matC      a dense matrix
* ldc       the size of the main dimension of the matrix C  
* output:
* matC      a dense matrix
*
*/

static openspblas_sparse_status_t (*spmmd_csr_layout_operation[])(const OPENSPBLAS_SPMAT_CSR *matA,
                                                            const OPENSPBLAS_SPMAT_CSR *matB,
                                                            OPENSPBLAS_Number *matC,
                                                            const OPENSPBLAS_INT ldc) = {
    spmmd_csr_row,
    spmmd_csr_col,
    spmmd_csr_row_trans,
    spmmd_csr_col_trans,
#ifdef COMPLEX
    spmmd_csr_row_conj, 
    spmmd_csr_col_conj, 
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* spmmd_csc_row          General row major matrics defined in csc storage format
* spmmd_csc_col          General column major matrics defined in csc storage format 
* spmmd_csc_row_trans    Transpose of general column major matrics defined in csc storage format
* spmmd_csc_col_trans    Transpose of general column major matrics defined in csc storage format
* spmmd_csc_row_conj     Conjugate transpose of general column major matrics defined in csc storage format
* spmmd_csc_col_conj     Conjugate transpose of general column major matrics defined in csc storage format 
* C := op(A)*B
*
* A         a sparse matrix
* B         a sparse matrix
* C         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* matA      a sparse matrix 
* matB      a sparse matrix
* matC      a dense matrix
* ldc       the size of the main dimension of the matrix C  
* output:
* matC      a dense matrix
*
*/

static openspblas_sparse_status_t (*spmmd_csc_layout_operation[])(const OPENSPBLAS_SPMAT_CSC *matA,
                                                           const OPENSPBLAS_SPMAT_CSC *matB,
                                                           OPENSPBLAS_Number *matC,
                                                           const OPENSPBLAS_INT ldc) = {
    spmmd_csc_row,
    spmmd_csc_col,
    spmmd_csc_row_trans,
    spmmd_csc_col_trans,
#ifdef COMPLEX
    spmmd_csc_row_conj, 
    spmmd_csc_col_conj, 
#endif
};

/*
* 
* Compute the dot product of a sparse matrix with a matrix
*
* details:
* spmmd_coo_row          General row major matrics defined in coo storage format
* spmmd_coo_col          General column major matrics defined in coo storage format 
* spmmd_coo_row_trans    Transpose of general column major matrics defined in coo storage format
* spmmd_coo_col_trans    Transpose of general column major matrics defined in coo storage format
* spmmd_coo_row_conj     Conjugate transpose of general column major matrics defined in coo storage format
* spmmd_coo_col_conj     Conjugate transpose of general column major matrics defined in coo storage format 
* C := op(A)*B
*
* A         a sparse matrix
* B         a sparse matrix
* C         a dense matrix
* op(A)     Data structure of sparse matrix
*
* input:
* matA      a sparse matrix 
* matB      a sparse matrix
* matC      a dense matrix
* ldc       the size of the main dimension of the matrix C  
* output:
* matC      a dense matrix
*
*/

static openspblas_sparse_status_t (*spmmd_bsr_layout_operation[])(const OPENSPBLAS_SPMAT_BSR *matA,
                                                           const OPENSPBLAS_SPMAT_BSR *matB,
                                                           OPENSPBLAS_Number *matC,
                                                           const OPENSPBLAS_INT ldc) = {
    spmmd_bsr_row,
    spmmd_bsr_col,
    spmmd_bsr_row_trans,
    spmmd_bsr_col_trans,
#ifdef COMPLEX
    spmmd_bsr_row_conj, 
    spmmd_bsr_col_conj, 
#endif
};

openspblas_sparse_status_t ONAME(const openspblas_sparse_operation_t operation,
                                             const openspblas_sparse_matrix_t A,
                                             const openspblas_sparse_matrix_t B,
                                             const openspblas_sparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             OPENSPBLAS_Number *matC,
                                             const OPENSPBLAS_INT ldc)
{
    check_null_return(A->mat, OPENSPBLAS_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(B->mat, OPENSPBLAS_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(matC, OPENSPBLAS_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->format != B->format, OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE);
    check_return(A->datatype != OPENSPBLAS_SPARSE_DATATYPE, OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE);
    check_return(B->datatype != OPENSPBLAS_SPARSE_DATATYPE, OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
    if(operation == OPENSPBLAS_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE;
#endif

    check_return(!check_equal_colA_rowB(A, B, operation), OPENSPBLAS_SPARSE_STATUS_INVALID_VALUE);

    if(A->format == OPENSPBLAS_SPARSE_FORMAT_CSR)
    {
        check_null_return(spmmd_csr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
        return spmmd_csr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](A->mat, B->mat, matC, ldc);
    }
    else if(A->format == OPENSPBLAS_SPARSE_FORMAT_CSC)
    {
        check_null_return(spmmd_csc_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
        return spmmd_csc_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](A->mat, B->mat, matC, ldc);
    }
    else if(A->format == OPENSPBLAS_SPARSE_FORMAT_BSR)
    {
        check_null_return(spmmd_bsr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)], OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED);
        return spmmd_bsr_layout_operation[index2(operation, layout, OPENSPBLAS_SPARSE_LAYOUT_NUM)](A->mat, B->mat, matC, ldc);
    }
    else
        return OPENSPBLAS_SPARSE_STATUS_NOT_SUPPORTED;
}
