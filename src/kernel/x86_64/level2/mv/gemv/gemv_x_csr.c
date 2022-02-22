#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef S
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined (__AVX512__) && defined(S)
static float gemv_csr_kernel_doti_simd512_unroll4(const ALPHA_INT ns,const float* x,const ALPHA_INT* indx,const float* y){
   ALPHA_INT ns64 = (ns >> 6) << 6; 
   __m512 tmp0,tmp1,tmp2,tmp3;
   __m512 vy0,vy1,vy2,vy3;
   __m512 vx0,vx1,vx2,vx3;
   __m512i vindex0,vindex1,vindex2,vindex3;
   float tmp = 0;
   tmp0 = _mm512_setzero();
   tmp1 = _mm512_setzero();
   tmp2 = _mm512_setzero();
   tmp3 = _mm512_setzero();
   ALPHA_INT i;
   for(i = 0; i < ns64;i+=64){
       vx0 = _mm512_loadu_ps(&x[i]);
       vx1 = _mm512_loadu_ps(&x[i+16]);
       vx2 = _mm512_loadu_ps(&x[i+32]);
       vx3 = _mm512_loadu_ps(&x[i+48]);
       vindex0 = _mm512_loadu_si512((void *)&indx[i]);
       vindex1 = _mm512_loadu_si512((void *)&indx[i+16]);
       vindex2 = _mm512_loadu_si512((void *)&indx[i+32]);
       vindex3 = _mm512_loadu_si512((void *)&indx[i+48]);
       vy0 = _mm512_i32gather_ps(vindex0,y,4);
       vy1 = _mm512_i32gather_ps(vindex1,y,4);
       vy2 = _mm512_i32gather_ps(vindex2,y,4);
       vy3 = _mm512_i32gather_ps(vindex3,y,4);
       tmp0 = _mm512_fmadd_ps(vx0,vy0,tmp0);
       tmp1 = _mm512_fmadd_ps(vx1,vy1,tmp1);
       tmp2 = _mm512_fmadd_ps(vx2,vy2,tmp2);
       tmp3 = _mm512_fmadd_ps(vx3,vy3,tmp3);
   }   
   for(; i< ns;++i){
      tmp += x[i] * y[indx[i]]; 
   }
   tmp += _mm512_reduce_add_ps(tmp0) + _mm512_reduce_add_ps(tmp1) + _mm512_reduce_add_ps(tmp2) + _mm512_reduce_add_ps(tmp3);
   return tmp;
}
#endif
//
//static float gemv_s_csr_kernel_doti_simd512(const ALPHA_INT ns,const float* x,const ALPHA_INT* indx,const float* y){
//    ALPHA_INT ns16 = (ns >> 4) << 4; 
//    __m512 tmp,vy,vx;
//    __m512i vindex;
//    float tmp0 = 0;
//    tmp = _mm512_setzero();
//    ALPHA_INT i;
//    for(i = 0; i < ns16;i+=16){
//        vx = _mm512_loadu_ps(&x[i]);
//        vindex = _mm512_loadu_epi32(&indx[i]);
//    	vy = _mm512_i32gather_ps(vindex,y,4);
//        tmp = _mm512_fmadd_ps(vx,vy,tmp);
//    }   
//    for(; i< ns;++i){
//       tmp0 += x[i] * y[indx[i]]; 
//    }
//    tmp0 += _mm512_reduce_add_ps(tmp);
//    return tmp0;
//}

// static float gemv_s_csr_kernel_doti_unroll4(const ALPHA_INT ns,const float* x,const ALPHA_INT* indx,const float* y){
//     ALPHA_INT ns4 = ns & ~3; 
//     ALPHA_INT i;
//     float tmp0 = 0.f;
//     float tmp1 = 0.f;
//     float tmp2 = 0.f;
//     float tmp3 = 0.f;
//     for(i = 0; i < ns4;i+=4){
//         tmp0 += x[i] * y[indx[i]]; 
//         tmp1 += x[i+1] * y[indx[i+1]]; 
//         tmp2 += x[i+2] * y[indx[i+2]]; 
//         tmp3 += x[i+3] * y[indx[i+3]]; 
//     }   
//     for(; i< ns;++i){
//        tmp0 += x[i] * y[indx[i]]; 
//     }   
//     return ((tmp0 + tmp1) + (tmp2 + tmp3));
// }

// float gemv_s_csr_kernel_doti(const ALPHA_INT ns,const float* x,const ALPHA_INT* indx,const float* y){
//     float tmp0 = 0.f;
//     for(ALPHA_INT i = 0; i< ns;++i){
//        tmp0 += x[i] * y[indx[i]]; 
//     }   
//     return tmp0;
// }

// alphasparse_status_t gemv_s_csr(const float alpha,const spmat_csr_s_t* A,const float* x,const float beta,float* y)
// {
//     ALPHA_INT m = A->rows;
//     ALPHA_INT num_threads = alpha_get_thread_num();
//     ALPHA_INT partition[num_threads + 1];
//     balanced_partition_row_by_nnz(A->rows_end, m, num_threads, partition);

// #ifdef _OPENMP
// #pragma omp parallel num_threads(num_threads)
// #endif
//     {
//         ALPHA_INT tid = alpha_get_thread_id();

//         ALPHA_INT local_m_s = partition[tid];
//         ALPHA_INT local_m_e = partition[tid + 1];

//         for (ALPHA_INT i = local_m_s; i < local_m_e; i++)
//         {
//             y[i] *= beta;
//             ALPHA_INT pks = A->rows_start[i];
//             ALPHA_INT pke = A->rows_end[i];
//             ALPHA_INT pkl = pke - pks;
//             // float tmp = gemv_s_csr_kernel_doti(pkl,&A->values[pks],&A->col_indx[pks],x);
//             //float tmp = gemv_s_csr_kernel_doti_unroll4(pkl,&A->values[pks],&A->col_indx[pks],x);
//             //float tmp = gemv_s_csr_kernel_doti_simd512(pkl,&A->values[pks],&A->col_indx[pks],x);
//             float tmp = gemv_s_csr_kernel_doti_simd512_unroll4(pkl,&A->values[pks],&A->col_indx[pks],x);
//             y[i] += alpha * tmp;
//         }
//     }
//     return ALPHA_SPARSE_STATUS_SUCCESS;
// }

static ALPHA_Number gemv_kernel_doti_unroll4(const ALPHA_INT ns, const ALPHA_Number *x, const ALPHA_INT *indx, const ALPHA_Number *y)
{
    ALPHA_INT ns4 = ((ns >> 2) << 2);
    ALPHA_INT i;
    ALPHA_Number tmp0, tmp1, tmp2, tmp3;
    alpha_setzero(tmp0);
    alpha_setzero(tmp1);
    alpha_setzero(tmp2);
    alpha_setzero(tmp3);
    for (i = 0; i < ns4; i += 4)
    {
        alpha_madde(tmp0, x[i], y[indx[i]]);
        alpha_madde(tmp1, x[i + 1], y[indx[i + 1]]);
        alpha_madde(tmp2, x[i + 2], y[indx[i + 2]]);
        alpha_madde(tmp3, x[i + 3], y[indx[i + 3]]);
    }
    for (; i < ns; ++i)
    {
        alpha_madde(tmp0, x[i], y[indx[i]]);
    }
    alpha_adde(tmp0, tmp1);
    alpha_adde(tmp2, tmp3);
    alpha_adde(tmp0, tmp2);
    return tmp0;
}

static alphasparse_status_t
gemv_csr_unroll4(const ALPHA_Number alpha,
                   const ALPHA_SPMAT_CSR *A,
                   const ALPHA_Number *x,
                   const ALPHA_Number beta,
                   ALPHA_Number *y,
                   ALPHA_INT lrs,
                   ALPHA_INT lre)
{
    for (ALPHA_INT i = lrs; i < lre; i++)
    {
        ALPHA_INT pks = A->rows_start[i];
        ALPHA_INT pke = A->rows_end[i];
        ALPHA_INT pkl = pke - pks;

#if defined (__AVX512__) && defined(S)
        float tmp = gemv_csr_kernel_doti_simd512_unroll4(pkl,&A->values[pks],&A->col_indx[pks],x); 
#else
        ALPHA_Number tmp = gemv_kernel_doti_unroll4(pkl, &A->values[pks], &A->col_indx[pks], x);
#endif
// #else
//         ALPHA_Number tmp = gemv_kernel_doti_unroll4(pkl, &A->values[pks], &A->col_indx[pks], x);
// #endif
        alpha_mule(y[i], beta);
        alpha_madde(y[i], alpha, tmp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

static alphasparse_status_t
gemv_csr_omp(const ALPHA_Number alpha,
               const ALPHA_SPMAT_CSR *A,
               const ALPHA_Number *x,
               const ALPHA_Number beta,
               ALPHA_Number *y)
{
    ALPHA_INT m = A->rows;

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(A->rows_end, m, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();

        ALPHA_INT local_m_s = partition[tid];
        ALPHA_INT local_m_e = partition[tid + 1];
        gemv_csr_unroll4(alpha, A, x, beta, y, local_m_s, local_m_e);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t
ONAME(const ALPHA_Number alpha,
      const ALPHA_SPMAT_CSR *mat,
      const ALPHA_Number *x,
      const ALPHA_Number beta,
      ALPHA_Number *y)
{
    return gemv_csr_omp(alpha, mat, x, beta, y);
}
