#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <stdbool.h>
#include <memory.h>

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSC *A, const ALPHA_SPMAT_CSC *B, ALPHA_SPMAT_CSC **matC)
{
    // 稀疏矩阵A * 稀疏矩阵B -> 稀疏矩阵matC
    check_return(A->cols != B->rows, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    ALPHA_SPMAT_CSC *mat = alpha_malloc(sizeof(ALPHA_SPMAT_CSC));
    *matC = mat;
    mat->rows = A->rows;
    mat->cols = B->cols;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    // 计算所需空间（传过来的matC是没有分配空间的）
    bool *flag = alpha_memalign(sizeof(bool) * m, DEFAULT_ALIGNMENT);
    ALPHA_INT nnz = 0;
    for (ALPHA_INT bc = 0; bc < n; bc++)
    {
        memset(flag, '\0', sizeof(bool) * m);
        for (ALPHA_INT bi = B->cols_start[bc]; bi < B->cols_end[bc]; bi++)
        {
            ALPHA_INT ac = B->row_indx[bi];
            for (ALPHA_INT ai = A->cols_start[ac]; ai < A->cols_end[ac]; ai++)
            {
                if (!flag[A->row_indx[ai]])
                {
                    nnz += 1;
                    flag[A->row_indx[ai]] = true;
                }
            }
        }
    }
    alpha_free(flag);
    //printf("%d", nnz);

    ALPHA_INT *col_offset = alpha_memalign(sizeof(ALPHA_INT) * (n + 1), DEFAULT_ALIGNMENT);
    mat->cols_start = col_offset;
    mat->cols_end = col_offset + 1;
    mat->row_indx = alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->values = alpha_memalign(nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    memset(mat->values, '\0', sizeof(ALPHA_Number) * nnz); 

    ALPHA_Number *values = alpha_memalign(sizeof(ALPHA_Number) * m, DEFAULT_ALIGNMENT);
    bool *write_back = alpha_memalign(sizeof(bool) * m, DEFAULT_ALIGNMENT);

    ALPHA_INT index = 0;
    mat->cols_start[0] = 0;
    for (ALPHA_INT bc = 0; bc < n; bc++) //遍历B的列
    {
        memset(values, '\0', sizeof(ALPHA_Number) * m); //matC的每一列有m个元素
        memset(write_back, '\0', sizeof(bool) * m);
        for (ALPHA_INT bi = B->cols_start[bc]; bi < B->cols_end[bc]; bi++) //对B的第bc列做矩阵乘法
        {
            ALPHA_INT ac = B->row_indx[bi];
            ALPHA_Number bv;
            bv = B->values[bi];
            for (ALPHA_INT ai = A->cols_start[ac]; ai < A->cols_end[ac]; ai++)
            {
                ALPHA_INT ar = A->row_indx[ai];
                //values[ar] += bv * A->values[ai];
                ALPHA_Number av = A->values[ai];
                alpha_madde(values[ar], bv, av);
                write_back[ar] = true;
            }
        }
        for (ALPHA_INT r = 0; r < m; r++)// 把matC的第ar行的非零元素编码进csr
        {
            if (write_back[r])
            {
                mat->row_indx[index] = r;
                //mat->values[index] = values[r];
                mat->values[index] = values[r];
                //mat->values[index] = values[r];
                index += 1;
            }
        }
        mat->cols_end[bc] = index;
    }

    alpha_free(values);
    alpha_free(write_back);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
