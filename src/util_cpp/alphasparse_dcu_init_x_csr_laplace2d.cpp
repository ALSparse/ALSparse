// #include "alphasparse/handle.h"
// #include <hip/hip_runtime.h>

// #ifdef __cplusplus
// extern "C"
// {
// #endif /* __cplusplus */

// #include "alphasparse/spdef.h"
// #include "alphasparse/types.h"
// #include "alphasparse/util/error.h"
// #include "alphasparse/util/malloc.h"
// #include <math.h>
// #include <assert.h>
// #include <memory.h>

// void alphasparse_dcu_init_x_csr_laplace2d(ALPHA_INT*      row_ptr,
//                                   ALPHA_INT*      col_ind,
//                                   ALPHA_Number*   val,
//                                   ALPHA_INT              dim_x,
//                                   ALPHA_INT              dim_y,
//                                   ALPHA_INT&             M,
//                                   ALPHA_INT&             N,
//                                   ALPHA_INT&             nnz,
//                                   alphasparse_index_base_t base)
// {
//     // Do nothing
//     if(dim_x == 0 || dim_y == 0)
//     {
//         return;
//     }

//     M = dim_x * dim_y;
//     N = dim_x * dim_y;

//     // Approximate 9pt stencil
//     ALPHA_INT nnz_mat = 9 * M;

//     row_ptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (M + 1));
//     col_ind = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * nnz_mat);
//     val = (ALPHA_Number *)alpha_malloc(sizeof(ALPHA_Number) * nnz_mat);

//     nnz        = base;
//     row_ptr[0] = base;

//     // Fill local arrays
// #ifdef _OPENMP
// #pragma omp parallel for schedule(dynamic, 1024)
// #endif
//     for(int32_t iy = 0; iy < dim_y; ++iy)
//     {
//         for(int32_t ix = 0; ix < dim_x; ++ix)
//         {
//             J row = iy * dim_x + ix;

//             for(int32_t sy = -1; sy <= 1; ++sy)
//             {
//                 if(iy + sy > -1 && iy + sy < dim_y)
//                 {
//                     for(int32_t sx = -1; sx <= 1; ++sx)
//                     {
//                         if(ix + sx > -1 && ix + sx < dim_x)
//                         {
//                             J col = row + sy * dim_x + sx;

//                             col_ind[nnz - base] = col + base;
//                             val[nnz - base]     = (col == row) ? 8.0 : -1.0;

//                             ++nnz;
//                         }
//                     }
//                 }
//             }

//             row_ptr[row + 1] = nnz;
//         }
//     }

//     // Adjust nnz by index base
//     nnz -= base;
// }

// #ifdef __cplusplus
// }
// #endif /*__cplusplus */