#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *mat, const ALPHA_Number *x, const ALPHA_INT columns, const ALPHA_INT ldx, const ALPHA_Number beta, ALPHA_Number *y, const ALPHA_INT ldy)
{
    // 稀疏矩阵乘以稠密矩阵
    // y := alpha*A*x + beta*y
    ALPHA_INT rowA = mat->rows;
    ALPHA_INT colA = mat->cols;
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

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            //y[index2(cc, cr, ldy)] = beta * y[index2(cc, cr, ldy)] + alpha * diag[cr] * x[index2(cc, cr, ldx)];
            ALPHA_Number temp1, temp2;
            alpha_mul(temp1, beta, y[index2(cc, cr, ldy)]);
            alpha_mul(temp2, diag[cr], x[index2(cc, cr, ldx)]);
            alpha_mul(temp2, alpha, temp2);
            alpha_add(y[index2(cc, cr, ldy)], temp1, temp2);
        }
            
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
