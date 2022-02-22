#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "alphasparse/spmat.h"

static alphasparse_status_t (*add_csr_operation_plain[])(const ALPHA_SPMAT_CSR *A,
                                             const ALPHA_Number alpha,
                                             const ALPHA_SPMAT_CSR *B,
                                             ALPHA_SPMAT_CSR **C) = {
    add_csr_plain,
    add_csr_trans_plain,
    NULL, // add_csr_conj_plain,
};

alphasparse_status_t ONAME(const alphasparse_operation_t operation,
                                     const alphasparse_matrix_t A,
                                     const ALPHA_Number alpha,
                                     const alphasparse_matrix_t B,
                                     alphasparse_matrix_t *matC)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(B->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(matC, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(B->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(A->format != B->format, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    check_return(A->format != ALPHA_SPARSE_FORMAT_CSR, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);

#ifndef COMPLEX
    if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
#endif

    alphasparse_matrix *AA = alpha_malloc(sizeof(alphasparse_matrix));
    *matC = AA;
    ALPHA_SPMAT_CSR *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSR));
    AA->format = A->format;
    AA->datatype = A->datatype;
    AA->mat = mat;

    check_null_return(add_csr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    return add_csr_operation_plain[operation]((const ALPHA_SPMAT_CSR *)A->mat, alpha, (const ALPHA_SPMAT_CSR *)B->mat, (ALPHA_SPMAT_CSR **)&AA->mat);
}