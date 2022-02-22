#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_Number alpha, const ALPHA_SPMAT_CSC *A, const ALPHA_Number *x, ALPHA_Number *y)
{
    //ALPHA_Number alphax[A->cols]; //为了存储alph*x的结果，由于用一个ac的y在被反复使用，只能用临时数组存储
    //memset(alphax, '\0', A->cols * sizeof(ALPHA_Number));
    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        //alphax[c] = alpha * x[c];  //给alphax赋初值
        //y[c] = alpha *x[c];
        alpha_mul(y[c], alpha, x[c]);
    }
    for (ALPHA_INT ac = A->cols - 1; ac >= 0; ac--) //遍历A的每一列
    {
        for (ALPHA_INT ai = A->cols_start[ac]; ai < A->cols_end[ac]; ai++) //遍历A的ac列中非零元素在values和row_indx的位置
        {
            ALPHA_INT ar = A->row_indx[ai];
            //ALPHA_Number val = A->values[ai];
            ALPHA_Number val;
            val = A->values[ai];
            if (ac > ar) // 访问上三角部分，不取=
            {
                //alphax[ar] -= val * y[ac];
                //y[ar] -= val * y[ac];
                ALPHA_Number t;
                alpha_mul(t, val, y[ac]);
                alpha_sub(y[ar], y[ar], t);
            }
        }
        //y[ac] = alphax[ac];
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
