#include "alphasparse/format.h"
#include "alphasparse/spapi.h"
#include "alphasparse/spmat.h"
#include "alphasparse/util/check.h"
#include "alphasparse/util/malloc.h"

// typedef struct
// {
//     float *values;
//     ALPHA_INT *row_indx;
//     ALPHA_INT *col_indx;
//     ALPHA_INT rows;
//     ALPHA_INT cols;
//     ALPHA_INT nnz;
// } spmat_coo_s_t;

alphasparse_status_t convert_coo_datatype_coo(const alpha_internal_spmat *source,
                                             alpha_internal_spmat **dest,
                                             alphasparse_datatype_t datatype) {
  // alpha_dump_feature_crd(source_coo->row_indx, source_coo->col_indx, source_coo->rows,
  // source_coo->cols, source_coo->nnz, Feature);

  // return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    spmat_coo_s_t *source_coo = (spmat_coo_s_t *)source;
    spmat_coo_s_t *dest_ = (spmat_coo_s_t *)alpha_malloc(sizeof(spmat_coo_s_t));

    *dest = (void *)dest_;
    dest_->row_indx = source_coo->row_indx;
    dest_->col_indx = source_coo->col_indx;
    dest_->rows = source_coo->rows;
    dest_->cols = source_coo->cols;
    dest_->values = source_coo->values;
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    spmat_coo_d_t *source_coo = (spmat_coo_d_t *)source;
    spmat_coo_d_t *dest_ = (spmat_coo_d_t *)alpha_malloc(sizeof(spmat_coo_d_t));

    *dest = (void *)dest_;
    dest_->row_indx = source_coo->row_indx;
    dest_->col_indx = source_coo->col_indx;
    dest_->rows = source_coo->rows;
    dest_->cols = source_coo->cols;
    dest_->values = source_coo->values;
  } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    spmat_coo_c_t *source_coo = (spmat_coo_c_t *)source;
    spmat_coo_c_t *dest_ = (spmat_coo_c_t *)alpha_malloc(sizeof(spmat_coo_c_t));

    *dest = (void *)dest_;
    dest_->row_indx = source_coo->row_indx;
    dest_->col_indx = source_coo->col_indx;
    dest_->rows = source_coo->rows;
    dest_->cols = source_coo->cols;
    dest_->values = source_coo->values;
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    spmat_coo_z_t *source_coo = (spmat_coo_z_t *)source;
    spmat_coo_z_t *dest_ = (spmat_coo_z_t *)alpha_malloc(sizeof(spmat_coo_z_t));

    *dest = (void *)dest_;
    dest_->row_indx = source_coo->row_indx;
    dest_->col_indx = source_coo->col_indx;
    dest_->rows = source_coo->rows;
    dest_->cols = source_coo->cols;
    dest_->values = source_coo->values;
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t convert_coo_datatype_csr(const alpha_internal_spmat *source,
                                             alpha_internal_spmat **dest,
                                             alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_coo_s_csr((spmat_csr_s_t *)source, (spmat_coo_s_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_coo_d_csr((spmat_csr_d_t *)source, (spmat_coo_d_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    return convert_coo_c_csr((spmat_csr_c_t *)source, (spmat_coo_c_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    return convert_coo_z_csr((spmat_csr_z_t *)source, (spmat_coo_z_t **)dest);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

// alphasparse_status_t convert_coo_datatype_coo(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return coo_s_copy((spmat_coo_s_t *)source, (spmat_coo_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return coo_d_copy((spmat_coo_d_t *)source, (spmat_coo_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return coo_c_copy((spmat_coo_c_t *)source, (spmat_coo_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return coo_z_copy((spmat_coo_z_t *)source, (spmat_coo_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparse_status_t convert_coo_datatype_bsr(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_coo_s_bsr((spmat_bsr_s_t *)source, (spmat_coo_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_coo_d_bsr((spmat_bsr_d_t *)source, (spmat_coo_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_coo_c_bsr((spmat_bsr_c_t *)source, (spmat_coo_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_coo_z_bsr((spmat_bsr_z_t *)source, (spmat_coo_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

alphasparse_status_t convert_coo_datatype_format(const alpha_internal_spmat *source,
                                                alpha_internal_spmat **dest,
                                                alphasparse_datatype_t datatype,
                                                alphasparse_format_t format) {
  if (format == ALPHA_SPARSE_FORMAT_COO) {
    return convert_coo_datatype_coo(source, dest, datatype);
  } else if (format == ALPHA_SPARSE_FORMAT_CSR) {
    return convert_coo_datatype_csr(source, dest, datatype);
  }
  // else if (format == ALPHA_SPARSE_FORMAT_CSC)
  // {
  //     return convert_coo_datatype_coo(source, dest, datatype);
  // }
  // else if (format == ALPHA_SPARSE_FORMAT_BSR)
  // {
  //     return convert_coo_datatype_bsr(source, dest, datatype);
  // }
  else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t alphasparse_convert_coo(
    const alphasparse_matrix_t source,       /* convert original matrix to CSC representation */
    const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  if (source->format != ALPHA_SPARSE_FORMAT_COO && source->format != ALPHA_SPARSE_FORMAT_CSR) {
    *dest = NULL;
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  alphasparse_matrix *dest_ = alpha_malloc(sizeof(alphasparse_matrix));
  *dest = dest_;
  dest_->dcu_info = NULL;
  dest_->format = ALPHA_SPARSE_FORMAT_COO;
  dest_->datatype = source->datatype;
  alphasparse_status_t status;

  if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    return convert_coo_datatype_format((const alpha_internal_spmat *)source->mat,
                                       (alpha_internal_spmat **)&dest_->mat, source->datatype,
                                       source->format);
  } else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    alphasparse_matrix_t AA;
    check_error_return(alphasparse_transpose(source, &AA));
    status =
        convert_coo_datatype_format((const alpha_internal_spmat *)AA->mat,
                                    (alpha_internal_spmat **)&dest_->mat, AA->datatype, AA->format);
    alphasparse_destroy(AA);
    return status;
  } else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t alphasparse_convert_coo_internal(
    const alphasparse_matrix_t source,       /* convert original matrix to BSR representation */
    const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  return alphasparse_convert_coo(source, operation, dest);
}
