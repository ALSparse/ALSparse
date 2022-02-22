/**
 * @brief implement for alphasparse_?_spmm intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"

alphasparse_status_t (*spmm_s_csr_operation_plain[])(const spmat_csr_s_t *A,
                                                    const spmat_csr_s_t *B,
                                                    spmat_csr_s_t **C) = {

    spmm_s_csr_plain,
    spmm_s_csr_trans_plain,
    NULL, // spmm_s_csr_conj_plain, 
};

alphasparse_status_t (*spmm_d_csr_operation_plain[])(const spmat_csr_d_t *A,
                                                    const spmat_csr_d_t *B,
                                                    spmat_csr_d_t **C) = {

    spmm_d_csr_plain,
    spmm_d_csr_trans_plain,
    NULL, // spmm_d_csr_conj_plain, 
};

alphasparse_status_t (*spmm_c_csr_operation_plain[])(const spmat_csr_c_t *A,
                                                    const spmat_csr_c_t *B,
                                                    spmat_csr_c_t **C) = {

    spmm_c_csr_plain,
    spmm_c_csr_trans_plain,
    spmm_c_csr_conj_plain, 
};

alphasparse_status_t (*spmm_z_csr_operation_plain[])(const spmat_csr_z_t *A,
                                                    const spmat_csr_z_t *B,
                                                    spmat_csr_z_t **C) = {

    spmm_z_csr_plain,
    spmm_z_csr_trans_plain,
    spmm_z_csr_conj_plain, 
};

alphasparse_status_t (*spmm_s_csc_operation_plain[])(const spmat_csc_s_t *A,
                                                    const spmat_csc_s_t *B,
                                                    spmat_csc_s_t **C) = {

    spmm_s_csc_plain,
    spmm_s_csc_trans_plain,
    NULL, // spmm_s_csc_conj_plain, 
};

alphasparse_status_t (*spmm_d_csc_operation_plain[])(const spmat_csc_d_t *A,
                                                    const spmat_csc_d_t *B,
                                                    spmat_csc_d_t **C) = {

    spmm_d_csc_plain,
    spmm_d_csc_trans_plain,
    NULL, // spmm_d_csc_conj_plain, 
};

alphasparse_status_t (*spmm_c_csc_operation_plain[])(const spmat_csc_c_t *A,
                                                    const spmat_csc_c_t *B,
                                                    spmat_csc_c_t **C) = {

    spmm_c_csc_plain,
    spmm_c_csc_trans_plain,
    spmm_c_csc_conj_plain, 
};

alphasparse_status_t (*spmm_z_csc_operation_plain[])(const spmat_csc_z_t *A,
                                                    const spmat_csc_z_t *B,
                                                    spmat_csc_z_t **C) = {

    spmm_z_csc_plain,
    spmm_z_csc_trans_plain,
    spmm_z_csc_conj_plain, 
};

alphasparse_status_t (*spmm_s_bsr_operation_plain[])(const spmat_bsr_s_t *A,
                                                    const spmat_bsr_s_t *B,
                                                    spmat_bsr_s_t **C) = {

    spmm_s_bsr_plain,
    spmm_s_bsr_trans_plain,
    NULL, // spmm_s_bsr_conj_plain, 
};

alphasparse_status_t (*spmm_d_bsr_operation_plain[])(const spmat_bsr_d_t *A,
                                                    const spmat_bsr_d_t *B,
                                                    spmat_bsr_d_t **C) = {

    spmm_d_bsr_plain,
    spmm_d_bsr_trans_plain,
    NULL, // spmm_d_bsr_conj_plain, 
};

alphasparse_status_t (*spmm_c_bsr_operation_plain[])(const spmat_bsr_c_t *A,
                                                    const spmat_bsr_c_t *B,
                                                    spmat_bsr_c_t **C) = {

    spmm_c_bsr_plain,
    spmm_c_bsr_trans_plain,
    spmm_c_bsr_conj_plain, 
};

alphasparse_status_t (*spmm_z_bsr_operation_plain[])(const spmat_bsr_z_t *A,
                                                    const spmat_bsr_z_t *B,
                                                    spmat_bsr_z_t **C) = {

    spmm_z_bsr_plain,
    spmm_z_bsr_trans_plain,
    spmm_z_bsr_conj_plain, 
};

alphasparse_status_t alphasparse_spmm_plain(const alphasparse_operation_t operation, const alphasparse_matrix_t A, const alphasparse_matrix_t B, alphasparse_matrix_t *C)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(B->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(C, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->datatype != B->datatype, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(A->format != B->format, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    // check if colA == rowB
    check_return(!check_equal_colA_rowB(A, B, operation), ALPHA_SPARSE_STATUS_INVALID_VALUE);

    if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE && A->datatype <= ALPHA_SPARSE_DATATYPE_DOUBLE)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    alphasparse_matrix *CC = alpha_malloc(sizeof(alphasparse_matrix));
    *C = CC;

    CC->datatype = A->datatype;
    CC->format = A->format;

    if (A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
        {
            check_null_return(spmm_s_csr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_s_csr_operation_plain[operation]((const spmat_csr_s_t *)A->mat, (const spmat_csr_s_t *)B->mat, (spmat_csr_s_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
        {
            check_null_return(spmm_d_csr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_d_csr_operation_plain[operation]((const spmat_csr_d_t *)A->mat, (const spmat_csr_d_t *)B->mat, (spmat_csr_d_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        {
            check_null_return(spmm_c_csr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_c_csr_operation_plain[operation]((const spmat_csr_c_t *)A->mat, (const spmat_csr_c_t *)B->mat, (spmat_csr_c_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        {
            check_null_return(spmm_z_csr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_z_csr_operation_plain[operation]((const spmat_csr_z_t *)A->mat, (const spmat_csr_z_t *)B->mat, (spmat_csr_z_t **)&CC->mat);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
        {
            check_null_return(spmm_s_csc_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_s_csc_operation_plain[operation]((const spmat_csc_s_t *)A->mat, (const spmat_csc_s_t *)B->mat, (spmat_csc_s_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
        {
            check_null_return(spmm_d_csc_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_d_csc_operation_plain[operation]((const spmat_csc_d_t *)A->mat, (const spmat_csc_d_t *)B->mat, (spmat_csc_d_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        {
            check_null_return(spmm_c_csc_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_c_csc_operation_plain[operation]((const spmat_csc_c_t *)A->mat, (const spmat_csc_c_t *)B->mat, (spmat_csc_c_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        {
            check_null_return(spmm_z_csc_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_z_csc_operation_plain[operation]((const spmat_csc_z_t *)A->mat, (const spmat_csc_z_t *)B->mat, (spmat_csc_z_t **)&CC->mat);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
        {
            check_null_return(spmm_s_bsr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_s_bsr_operation_plain[operation]((const spmat_bsr_s_t *)A->mat, (const spmat_bsr_s_t *)B->mat, (spmat_bsr_s_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
        {
            check_null_return(spmm_d_bsr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_d_bsr_operation_plain[operation]((const spmat_bsr_d_t *)A->mat, (const spmat_bsr_d_t *)B->mat, (spmat_bsr_d_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        {
            check_null_return(spmm_c_bsr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_c_bsr_operation_plain[operation]((const spmat_bsr_c_t *)A->mat, (const spmat_bsr_c_t *)B->mat, (spmat_bsr_c_t **)&CC->mat);
        }
        else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        {
            check_null_return(spmm_z_bsr_operation_plain[operation], ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
            return spmm_z_bsr_operation_plain[operation]((const spmat_bsr_z_t *)A->mat, (const spmat_bsr_z_t *)B->mat, (spmat_bsr_z_t **)&CC->mat);
        }
        else
        {
            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}
