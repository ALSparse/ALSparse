/**
 * @brief implement for alphasparse_?_mm intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"

static alphasparse_status_t (*spmmd_csr_layout_operation_plain[])(const ALPHA_SPMAT_CSR *matA,
                                                            const ALPHA_SPMAT_CSR *matB,
                                                            ALPHA_Number *matC,
                                                            const ALPHA_INT ldc) = {
    spmmd_csr_row_plain,
    spmmd_csr_col_plain,
    spmmd_csr_row_trans_plain,
    spmmd_csr_col_trans_plain,
#ifdef COMPLEX
    spmmd_csr_row_conj_plain, 
    spmmd_csr_col_conj_plain, 
#endif
};

static alphasparse_status_t (*spmmd_csc_layout_operation_plain[])(const ALPHA_SPMAT_CSC *matA,
                                                            const ALPHA_SPMAT_CSC *matB,
                                                            ALPHA_Number *matC,
                                                            const ALPHA_INT ldc) = {
    spmmd_csc_row_plain,
    spmmd_csc_col_plain,
    spmmd_csc_row_trans_plain,
    spmmd_csc_col_trans_plain,
#ifdef COMPLEX
    spmmd_csc_row_conj_plain, 
    spmmd_csc_col_conj_plain, 
#endif
};

static alphasparse_status_t (*spmmd_bsr_layout_operation_plain[])(const ALPHA_SPMAT_BSR *matA,
                                                            const ALPHA_SPMAT_BSR *matB,
                                                            ALPHA_Number *matC,
                                                            const ALPHA_INT ldc) = {
    spmmd_bsr_row_plain,
    spmmd_bsr_col_plain,
    spmmd_bsr_row_trans_plain,
    spmmd_bsr_col_trans_plain,
#ifdef COMPLEX
    spmmd_bsr_row_conj_plain, 
    spmmd_bsr_col_conj_plain, 
#endif
};

alphasparse_status_t ONAME(const alphasparse_operation_t operation,
                                             const alphasparse_matrix_t A,
                                             const alphasparse_matrix_t B,
                                             const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                             ALPHA_Number *matC,
                                             const ALPHA_INT ldc)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(B->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(matC, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->format != B->format, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(B->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    // check if colA == rowB
    check_return(!check_equal_colA_rowB(A, B, operation), ALPHA_SPARSE_STATUS_INVALID_VALUE);

#ifndef COMPLEX
    if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif

    if(A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        check_null_return(spmmd_csr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        return spmmd_csr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](A->mat, B->mat, matC, ldc);
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        check_null_return(spmmd_csc_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        return spmmd_csc_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](A->mat, B->mat, matC, ldc);
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        check_null_return(spmmd_bsr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
        return spmmd_bsr_layout_operation_plain[index2(operation, layout, ALPHA_SPARSE_LAYOUT_NUM)](A->mat, B->mat, matC, ldc);
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}