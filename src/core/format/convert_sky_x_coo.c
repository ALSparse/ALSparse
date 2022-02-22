#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

static int row_first_cmp(const ALPHA_Point *a, const ALPHA_Point *b)
{
    if (a->x != b->x)
        return a->x - b->x;
    return a->y - b->y;
}

static int col_first_cmp(const ALPHA_Point *a, const ALPHA_Point *b)
{
    if (a->y != b->y)
        return a->y - b->y;
    return a->x - b->x;
}

alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_SKY **dest, const alphasparse_fill_mode_t fill){
    ALPHA_SPMAT_SKY* mat = alpha_malloc(sizeof(ALPHA_SPMAT_SKY)); 
    *dest = mat;
    mat->fill = fill;
    mat->rows = source->rows;
    mat->cols = source->cols;
    ALPHA_INT nnz = source->nnz;
    ALPHA_INT m = source->rows;
    ALPHA_INT n = source->cols;
    if(fill == ALPHA_SPARSE_FILL_MODE_LOWER){
        // sort by (row,col)
        ALPHA_Point *points = alpha_malloc(sizeof(ALPHA_Point) * nnz);
        ALPHA_INT count = 0;
        for (ALPHA_INT i = 0; i < nnz; i++)
        {
            if(source->row_indx[i] >= source->col_indx[i]){
                points[count].x = source->row_indx[i];
                points[count].y = source->col_indx[i];
                points[count].v = source->values[i];
                count+=1;
            }
        }
        qsort(points, count, sizeof(ALPHA_Point), (__compar_fn_t)row_first_cmp);
        //printf("%d\n",count);
        ALPHA_INT *rows_offset = alpha_memalign((m + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
        mat->pointers = rows_offset;
        mat->pointers[0] = 0;
        for(ALPHA_INT r = 0 , idx = 0;r < mat->rows;++r){
            while(idx < count && points[idx].x < r) ++idx;
            ALPHA_INT row_start = idx;
            if(idx == count){ // 已遍历完所有非零元
                mat->pointers[r+1] = mat->pointers[r]+1;
                continue;
            }
            if(points[idx].x == r){
                mat->pointers[r+1] = mat->pointers[r] + r - points[idx].y + 1;
            }else{   // 空行
                mat->pointers[r+1] = mat->pointers[r]+1;
            }
        } 
        ALPHA_INT sky_nnz = mat->pointers[m];
        mat->values = alpha_memalign(sky_nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
        memset(mat->values,'\0',sky_nnz * sizeof(ALPHA_Number));
        for(ALPHA_INT i = 0;i<count;++i){
            ALPHA_INT row = points[i].x;
            ALPHA_INT col = points[i].y;
            ALPHA_INT row_end = mat->pointers[row+1];
            mat->values[row_end-(row - col + 1)] = points[i].v;
        }
    }else if(fill == ALPHA_SPARSE_FILL_MODE_UPPER){
        // sort by (col,row)
        ALPHA_Point *points = alpha_malloc(sizeof(ALPHA_Point) * nnz);
        ALPHA_INT count = 0;
        for (ALPHA_INT i = 0; i < nnz; i++)
        {
            if(source->row_indx[i] <= source->col_indx[i]){
                points[count].x = source->row_indx[i];
                points[count].y = source->col_indx[i];
                points[count].v = source->values[i];
                count+=1;
            }
        }
        qsort(points, count, sizeof(ALPHA_Point), (__compar_fn_t)col_first_cmp);
        ALPHA_INT *cols_offset = alpha_memalign((n + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
        mat->pointers = cols_offset;
        mat->pointers[0] = 0;
        for(ALPHA_INT c = 0 , idx = 0;c < mat->cols;++c){
            while(idx < count && points[idx].y < c) ++idx;
            ALPHA_INT col_start = idx;
            if(idx == count){ // 已遍历完所有非零元
                mat->pointers[c+1] = mat->pointers[c]+1;
                continue;
            }
            if(points[idx].y == c){
                mat->pointers[c+1] = mat->pointers[c] + c - points[idx].x + 1;
            }else{   // 空行
                mat->pointers[c+1] = mat->pointers[c]+1;
            }
        } 
        ALPHA_INT sky_nnz = mat->pointers[n];
        mat->values = alpha_memalign(sky_nnz * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
        memset(mat->values,'\0',sky_nnz * sizeof(ALPHA_Number));
        for(ALPHA_INT i = 0;i<count;++i){
            ALPHA_INT row = points[i].x;
            ALPHA_INT col = points[i].y;
            ALPHA_INT col_end = mat->pointers[col+1];
            mat->values[col_end-(col - row + 1)] = points[i].v;
        }
    }else{
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
