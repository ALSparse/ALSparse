#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

alphasparse_status_t ONAME (alphasparse_matrix_t A, 
                        const ALPHA_INT nvalues, 
                        const ALPHA_INT *indx, 
                        const ALPHA_INT *indy, 
                        ALPHA_Number *values)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    // if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    // {
    //     return update_values_bsr_plain(A->mat, nvalues, indx, indy, values);
    // }
    // else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}