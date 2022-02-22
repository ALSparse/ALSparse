/**
 * @brief implement for alphasparse_destroy intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t destroy_datatype_coo(alpha_internal_spmat *mat, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return destroy_s_coo((spmat_coo_s_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return destroy_d_coo((spmat_coo_d_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return destroy_c_coo((spmat_coo_c_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return destroy_z_coo((spmat_coo_z_t *)mat);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t destroy_datatype_csr(alpha_internal_spmat *mat, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return destroy_s_csr((spmat_csr_s_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return destroy_d_csr((spmat_csr_d_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return destroy_c_csr((spmat_csr_c_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return destroy_z_csr((spmat_csr_z_t *)mat);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t destroy_datatype_csc(alpha_internal_spmat *mat, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return destroy_s_csc((spmat_csc_s_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return destroy_d_csc((spmat_csc_d_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return destroy_c_csc((spmat_csc_c_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return destroy_z_csc((spmat_csc_z_t *)mat);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t destroy_datatype_bsr(alpha_internal_spmat *mat, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return destroy_s_bsr((spmat_bsr_s_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return destroy_d_bsr((spmat_bsr_d_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return destroy_c_bsr((spmat_bsr_c_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return destroy_z_bsr((spmat_bsr_z_t *)mat);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t destroy_datatype_sky(alpha_internal_spmat *mat, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return destroy_s_sky((spmat_sky_s_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return destroy_d_sky((spmat_sky_d_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return destroy_c_sky((spmat_sky_c_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return destroy_z_sky((spmat_sky_z_t *)mat);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t destroy_datatype_dia(alpha_internal_spmat *mat, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return destroy_s_dia((spmat_dia_s_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return destroy_d_dia((spmat_dia_d_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return destroy_c_dia((spmat_dia_c_t *)mat);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return destroy_z_dia((spmat_dia_z_t *)mat);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t destroy_datatype_format(alpha_internal_spmat *mat, alphasparse_datatype_t datatype, alphasparse_format_t format)
{
    if (format == ALPHA_SPARSE_FORMAT_COO)
    {
        return destroy_datatype_coo(mat, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_CSR)
    {
        return destroy_datatype_csr(mat, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_CSC)
    {
        return destroy_datatype_csc(mat, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_BSR)
    {
        return destroy_datatype_bsr(mat, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_SKY)
    {
        return destroy_datatype_sky(mat, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_DIA)
    {
        return destroy_datatype_dia(mat, datatype);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t alphasparse_destroy(alphasparse_matrix_t A)
{
    check_null_return(A, ALPHA_SPARSE_STATUS_SUCCESS);
    if (A->mat != NULL)
    {
        destroy_datatype_format(A->mat, A->datatype, A->format);
    }
    alpha_free(A);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
