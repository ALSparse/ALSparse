#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    ALPHA_INT colA = mat->cols;
    ALPHA_INT rowC = mat->rows;
    ALPHA_INT colC = columns;
    ALPHA_Number diag[colA]; //存储对角元素
    for (ALPHA_INT ac = 0; ac < colA; ++ac) //将mat的对角元素提取出来，存入diag
    {
        alpha_setzero(diag[ac]);
        for (ALPHA_INT ai = mat->cols_start[ac]; ai < mat->cols_end[ac]; ++ai)
            if (mat->row_indx[ai] == ac)
            {
                //diag[ac] = mat->values[ai];
                diag[ac] = mat->values[ai];
            }
    }

    for (ALPHA_INT cr = 0; cr < rowC; ++cr)
        for (ALPHA_INT cc = 0; cc < colC; ++cc)
        {
            //y[index2(cr, cc, ldy)] = beta * y[index2(cr, cc, ldy)] + alpha * diag[cr] * x[index2(cr, cc, ldx)];
            ALPHA_Number temp1, temp2;
            alpha_mul(temp1, beta, y[index2(cr, cc, ldy)]);
            alpha_mul(temp2, diag[cr], x[index2(cr, cc, ldx)]);
            alpha_mul(temp2, alpha, temp2);
            alpha_add(y[index2(cr, cc, ldy)], temp1, temp2);
        }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
