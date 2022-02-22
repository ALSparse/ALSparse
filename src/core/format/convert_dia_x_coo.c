#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>


alphasparse_status_t ONAME(const ALPHA_SPMAT_COO *source, ALPHA_SPMAT_DIA **dest){
    ALPHA_SPMAT_DIA *mat = alpha_malloc(sizeof(ALPHA_SPMAT_DIA));
    *dest = mat;
    ALPHA_INT rows = source->rows;
    ALPHA_INT cols = source->cols;
    ALPHA_INT nnz = source->nnz;
    ALPHA_INT diag_num = rows + cols - 1;
    bool *flag = alpha_malloc(sizeof(bool)*diag_num);    
    memset(flag,'\0',sizeof(bool)*diag_num);
    for(ALPHA_INT i = 0;i < nnz;i++){
        ALPHA_INT row = source->row_indx[i];
        ALPHA_INT col = source->col_indx[i];
        ALPHA_INT diag = col - row + rows - 1;
        flag[diag] = 1;
    }
    mat->rows = rows;
    mat->cols = cols;
    mat->lval = rows;
    mat->ndiag = 0;   
    for(ALPHA_INT i = 0; i < diag_num;++i){
        if(flag[i] == 1){
            mat->ndiag += 1;
        }    
    }
    mat->distance = alpha_malloc(sizeof(ALPHA_INT)*mat->ndiag);
    for(ALPHA_INT i = 0,index = 0; i < diag_num;++i){
        if(flag[i] == 1){
            mat->distance[index] = i - rows + 1;
            index+=1;
        }    
    }
    alpha_free(flag);
    ALPHA_INT* diag_pos_map = alpha_malloc(sizeof(ALPHA_INT)*diag_num);;
    for(ALPHA_INT i = 0;i<diag_num;i++){
        diag_pos_map[i] = -1;
    }
    for(ALPHA_INT i = 0;i<mat->ndiag;i++){
        diag_pos_map[mat->distance[i]+rows-1] = i;
    }
    mat->values = alpha_malloc(sizeof(ALPHA_Number)*mat->ndiag*mat->lval);
    memset(mat->values,'\0',sizeof(ALPHA_Number)*mat->ndiag*mat->lval);
    for(ALPHA_INT i = 0;i < nnz;i++){
        ALPHA_INT row = source->row_indx[i];
        ALPHA_INT col = source->col_indx[i];
        ALPHA_INT diag = col - row + rows - 1;
        ALPHA_INT pos = diag_pos_map[diag];
        mat->values[index2(pos,row,mat->lval)] = source->values[i];
    }
    alpha_free(diag_pos_map);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
