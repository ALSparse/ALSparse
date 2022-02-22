#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <string.h>

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowA = mat->rows;
    ALPHA_INT nnz = mat->nnz;

    ALPHA_Number diag[rowA];
    
    for(int i = 0; i < rowA; i++)
    {
        alpha_setzero(diag[i]); 
    }
   
    for (ALPHA_INT ar = 0; ar < nnz; ++ar)
    {
        if (mat->col_indx[ar] == mat->row_indx[ar])
        {
            diag[mat->row_indx[ar]] = mat->values[ar];
        }
    }

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            ALPHA_Number t;
            alpha_setzero(t);
            alpha_mul(t, alpha, diag[cr]);
            alpha_mul(t, t, x[index2(cc, cr, ldx)]);
            alpha_mul(y[index2(cc, cr, ldy)], beta, y[index2(cc, cr, ldy)]);
            alpha_add(y[index2(cc, cr, ldy)], y[index2(cc, cr, ldy)], t);
            // y[index2(cc, cr, ldy)] = beta * y[index2(cc, cr, ldy)] + alpha * diag[cr] * x[index2(cc, cr, ldx)];
        }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
