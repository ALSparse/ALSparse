#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <alphasparse/compute.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_DIA *A, ALPHA_SPMAT_DIA **B)
{
    ALPHA_SPMAT_DIA* mat = alpha_malloc(sizeof(ALPHA_SPMAT_DIA));
    *B = mat;
    ALPHA_INT rowA = A->rows;
    ALPHA_INT colA = A->cols;
    ALPHA_INT ndiagA = A->ndiag;
    mat->rows = colA;
    mat->cols = rowA;
    mat->ndiag = ndiagA;
    mat->lval = mat->rows;
    mat->distance = alpha_malloc(sizeof(ALPHA_INT)*ndiagA);
    for(ALPHA_INT i = 0;i<ndiagA;++i){
        mat->distance[i] = 0-A->distance[ndiagA - i -1];
    }
    mat->values = alpha_malloc(sizeof(ALPHA_Number)*mat->ndiag*mat->lval);
    memset(mat->values,'\0',sizeof(ALPHA_Number)*mat->ndiag*mat->lval);
    for(ALPHA_INT adi = 0,bdi = ndiagA - 1;adi<ndiagA;++adi,--bdi){
        ALPHA_INT ad = A->distance[adi];
        ALPHA_INT bd = mat->distance[bdi];
        ALPHA_INT ars = alpha_max(0,-ad);
        ALPHA_INT brs = alpha_max(0,-bd);
        ALPHA_INT acs = alpha_max(0,ad);
        ALPHA_INT an = alpha_min(rowA - ars,colA - acs);
        for(ALPHA_INT j = 0;j<an;++j){
            alpha_conj(mat->values[index2(bdi,brs+j,mat->lval)], A->values[index2(adi,ars+j,A->lval)]);
        }
    }
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}
