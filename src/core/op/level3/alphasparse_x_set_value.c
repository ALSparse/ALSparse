#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME (alphasparse_matrix_t A, 
                        const ALPHA_INT row, 
                        const ALPHA_INT col,
                        const ALPHA_Number value)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    if(A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        return set_value_csr(A->mat, row, col, value);
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        return set_value_csc(A->mat, row, col, value);
    }
    // else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    // {
    //     return set_value_bsr(A->mat, row, col, value);
    // }
    else if(A->format == ALPHA_SPARSE_FORMAT_COO)
    {
        return set_value_coo(A->mat, row, col, value);
    }
    // else if(A->format == ALPHA_SPARSE_FORMAT_SKY)
    // {
    //     return set_value_sky(A->mat, row, col, value);
    // }
    // else if(A->format == ALPHA_SPARSE_FORMAT_DIA)
    // {
    //     return set_value_dia(A->mat, row, col, value);
    // }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}