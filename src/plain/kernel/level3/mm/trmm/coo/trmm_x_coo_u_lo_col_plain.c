#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t
ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_COO *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        ALPHA_Number ctmp;
        alpha_setzero(ctmp);
        ALPHA_INT r = 0;
        for (ALPHA_INT nn = 0; nn < mat->nnz; ++nn)
        {
            ALPHA_INT cr =  mat->row_indx[nn]; 
            while(cr >= r)
            {
                ALPHA_Number t;
                alpha_setzero(t);
                alpha_mul(y[index2(cc, r, ldy)], beta, y[index2(cc, r, ldy)]);
                alpha_mul(t, alpha, x[index2(cc, r, ldx)]);
                alpha_add(y[index2(cc, r, ldy)], y[index2(cc, r, ldy)], t);
                // y[index2(cc, r, ldy)] = beta * y[index2(cc, r, ldy)] + alpha * x[index2(cc, r, ldx)]; 
                r++;
            }   
            if(mat->col_indx[nn] < cr)    
            {    
                alpha_madde(ctmp, mat->values[nn], x[index2(cc, mat->col_indx[nn], ldx)]);
                // ctmp += mat->values[nn] * x[index2(cc, mat->col_indx[nn], ldx)];                 
            }
            if(nn + 1 < mat->nnz && cr != mat->row_indx[nn + 1])
            {
                alpha_madde(y[index2(cc, cr, ldy)], alpha, ctmp);
                // y[index2(cc, cr, ldy)] += alpha * ctmp;
                alpha_setzero(ctmp);
            }
            else if(nn + 1 == mat->nnz)
            {
                alpha_madde(y[index2(cc, cr, ldy)], alpha, ctmp);
                // y[index2(cc, cr, ldy)] += alpha * ctmp;
            }
        }
        while(mat->rows > r)
        {
            alpha_mul(y[index2(cc, r, ldy)], beta, y[index2(cc, r, ldy)]);
            // y[index2(cc, r, ldy)] = beta * y[index2(cc, r, ldy)];
            r++;
        } 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
