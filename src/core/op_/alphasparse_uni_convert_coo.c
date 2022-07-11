#include "alphasparse.h"

alphasparse_status_t alphasparse_uni_convert_coo(
    const alphasparse_matrix_t source,       /* convert original matrix to CSC representation */
    const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {

    alphasparse_status_t status =  alphasparse_convert_coo(source, operation, dest);

    if(source->exe != ALPHA_SPARSE_EXE_HOST)
    {
#ifdef __DCU__
        host2device_coo(*dest);
#endif
        return status;
    }
    else
        return status;
}