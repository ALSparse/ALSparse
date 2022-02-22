#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparse_status_t convert_dia_datatype_coo(const alpha_internal_spmat *source, alpha_internal_spmat **dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return convert_dia_s_coo((spmat_coo_s_t *)source, (spmat_dia_s_t **)dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return convert_dia_d_coo((spmat_coo_d_t *)source, (spmat_dia_d_t **)dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return convert_dia_c_coo((spmat_coo_c_t *)source, (spmat_dia_c_t **)dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return convert_dia_z_coo((spmat_coo_z_t *)source, (spmat_dia_z_t **)dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

//alphasparse_status_t convert_dia_datatype_csr(const alpha_internal_spmat *source, alpha_internal_spmat **dest, alphasparse_datatype_t datatype)
//{
//    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//    {
//        return convert_dia_s_csr((spmat_csr_s_t *)source, (spmat_dia_s_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//    {
//        return convert_dia_d_csr((spmat_csr_d_t *)source, (spmat_dia_d_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//    {
//        return convert_dia_c_csr((spmat_csr_c_t *)source, (spmat_dia_c_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//    {
//        return convert_dia_z_csr((spmat_csr_z_t *)source, (spmat_dia_z_t **)dest);
//    }
//    else
//    {
//        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//    }
//}

//alphasparse_status_t convert_dia_datatype_csc(const alpha_internal_spmat *source, alpha_internal_spmat **dest, alphasparse_datatype_t datatype)
//{
//    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//    {
//        return convert_dia_s_csc((spmat_csc_s_t *)source, (spmat_dia_s_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//    {
//        return convert_dia_d_csc((spmat_csc_d_t *)source, (spmat_dia_d_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//    {
//        return convert_dia_c_csc((spmat_csc_c_t *)source, (spmat_dia_c_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//    {
//        return convert_dia_z_csc((spmat_csc_z_t *)source, (spmat_dia_z_t **)dest);
//    }
//    else
//    {
//        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//    }
//}

//alphasparse_status_t convert_dia_datatype_bsr(const alpha_internal_spmat *source, alpha_internal_spmat **dest, alphasparse_datatype_t datatype)
//{
//    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//    {
//        return convert_dia_s_bsr((spmat_bsr_s_t *)source, (spmat_dia_s_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//    {
//        return convert_dia_d_bsr((spmat_bsr_d_t *)source, (spmat_dia_d_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//    {
//        return convert_dia_c_bsr((spmat_bsr_c_t *)source, (spmat_dia_c_t **)dest);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//    {
//        return convert_dia_z_bsr((spmat_bsr_z_t *)source, (spmat_dia_z_t **)dest);
//    }
//    else
//    {
//        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//    }
//}

alphasparse_status_t convert_dia_datatype_format(const alpha_internal_spmat *source, alpha_internal_spmat **dest, alphasparse_datatype_t datatype, alphasparse_format_t format)
{
    if (format == ALPHA_SPARSE_FORMAT_COO)
    {
        return convert_dia_datatype_coo(source, dest, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_CSR)
    {
        //return convert_dia_datatype_csr(source, dest, datatype);
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (format == ALPHA_SPARSE_FORMAT_CSC)
    {
        //return convert_dia_datatype_csc(source, dest, datatype);
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (format == ALPHA_SPARSE_FORMAT_BSR)
    {
        //return convert_dia_datatype_bsr(source, dest, datatype);
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparse_status_t alphasparse_convert_dia(const alphasparse_matrix_t source,       /* convert original matrix to CSR representation */
                                           const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
                                           alphasparse_matrix_t *dest)
{
    check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    alphasparse_matrix* dest_ = alpha_malloc(sizeof(alphasparse_matrix));
    *dest = dest_;
    dest_->format = ALPHA_SPARSE_FORMAT_DIA;
    dest_->datatype = source->datatype;

    alphasparse_status_t status;

    if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    {
        return convert_dia_datatype_format((const alpha_internal_spmat *)source->mat, (alpha_internal_spmat **)&dest_->mat, source->datatype, source->format);
    }
    else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
    {
        alphasparse_matrix_t AA;
        check_error_return(alphasparse_transpose(source, &AA));
        status = convert_dia_datatype_format((const alpha_internal_spmat *)AA->mat, (alpha_internal_spmat **)&dest_->mat, AA->datatype, AA->format);
        alphasparse_destroy(AA);
        return status;
    }
    else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}
