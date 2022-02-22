#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void pack_matrix_col2row_s(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const float *xp0 = &X[index2(c, 0, ldX)];
        const float *xp1 = &X[index2(c + 1, 0, ldX)];
        const float *xp2 = &X[index2(c + 2, 0, ldX)];
        const float *xp3 = &X[index2(c + 3, 0, ldX)];
        float *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = colX4 < 0 ? 0 : colX4; c < colX; c += 1)
    {
        const float *xp0 = &X[index2(c, 0, ldX)];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col_s(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = 8 > alpha_get_thread_num() ? alpha_get_thread_num() : 8;
    ALPHA_INT rowX4 = rowX - 15;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < rowX4; r += 16)
    {
        const float *xp0 = &X[index2(r, 0, ldX)];
        const float *xp1 = &X[index2(r + 1, 0, ldX)];
        const float *xp2 = &X[index2(r + 2, 0, ldX)];
        const float *xp3 = &X[index2(r + 3, 0, ldX)];
        const float *xp4 = &X[index2(r + 4, 0, ldX)];
        const float *xp5 = &X[index2(r + 5, 0, ldX)];
        const float *xp6 = &X[index2(r + 6, 0, ldX)];
        const float *xp7 = &X[index2(r + 7, 0, ldX)];
        const float *xp8 = &X[index2(r + 8, 0, ldX)];
        const float *xp9 = &X[index2(r + 9, 0, ldX)];
        const float *xp10 = &X[index2(r + 10, 0, ldX)];
        const float *xp11 = &X[index2(r + 11, 0, ldX)];
        const float *xp12 = &X[index2(r + 12, 0, ldX)];
        const float *xp13 = &X[index2(r + 13, 0, ldX)];
        const float *xp14 = &X[index2(r + 14, 0, ldX)];
        const float *xp15 = &X[index2(r + 15, 0, ldX)];
        float *yp = &Y[r];
        float *yp1 = yp + ldY;
        float *yp2 = yp1 + ldY;
        float *yp3 = yp2 + ldY;
        ALPHA_INT colX4 = colX - 3;
        ALPHA_INT c = 0;
        for (; c < colX4; c+=4)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            *(yp + 8) = *(xp8++);
            *(yp + 9) = *(xp9++);
            *(yp + 10) = *(xp10++);
            *(yp + 11) = *(xp11++);
            *(yp + 12) = *(xp12++);
            *(yp + 13) = *(xp13++);
            *(yp + 14) = *(xp14++);
            *(yp + 15) = *(xp15++);
            yp += 4*ldY;
            *yp1 = *(xp0++);
            *(yp1 + 1) = *(xp1++);
            *(yp1 + 2) = *(xp2++);
            *(yp1 + 3) = *(xp3++);
            *(yp1 + 4) = *(xp4++);
            *(yp1 + 5) = *(xp5++);
            *(yp1 + 6) = *(xp6++);
            *(yp1 + 7) = *(xp7++);
            *(yp1 + 8) = *(xp8++);
            *(yp1 + 9) = *(xp9++);
            *(yp1 + 10) = *(xp10++);
            *(yp1 + 11) = *(xp11++);
            *(yp1 + 12) = *(xp12++);
            *(yp1 + 13) = *(xp13++);
            *(yp1 + 14) = *(xp14++);
            *(yp1 + 15) = *(xp15++);
            yp1 += 4*ldY;

            *yp2 = *(xp0++);
            *(yp2 + 1) = *(xp1++);
            *(yp2 + 2) = *(xp2++);
            *(yp2 + 3) = *(xp3++);
            *(yp2 + 4) = *(xp4++);
            *(yp2 + 5) = *(xp5++);
            *(yp2 + 6) = *(xp6++);
            *(yp2 + 7) = *(xp7++);
            *(yp2 + 8) = *(xp8++);
            *(yp2 + 9) = *(xp9++);
            *(yp2 + 10) = *(xp10++);
            *(yp2 + 11) = *(xp11++);
            *(yp2 + 12) = *(xp12++);
            *(yp2 + 13) = *(xp13++);
            *(yp2 + 14) = *(xp14++);
            *(yp2 + 15) = *(xp15++);
            yp2 += 4*ldY;

            *yp3 = *(xp0++);
            *(yp3 + 1) = *(xp1++);
            *(yp3 + 2) = *(xp2++);
            *(yp3 + 3) = *(xp3++);
            *(yp3 + 4) = *(xp4++);
            *(yp3 + 5) = *(xp5++);
            *(yp3 + 6) = *(xp6++);
            *(yp3 + 7) = *(xp7++);
            *(yp3 + 8) = *(xp8++);
            *(yp3 + 9) = *(xp9++);
            *(yp3 + 10) = *(xp10++);
            *(yp3 + 11) = *(xp11++);
            *(yp3 + 12) = *(xp12++);
            *(yp3 + 13) = *(xp13++);
            *(yp3 + 14) = *(xp14++);
            *(yp3 + 15) = *(xp15++);
            yp3 += 4*ldY;
        }
        for (; c < colX; c++)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            *(yp + 8) = *(xp8++);
            *(yp + 9) = *(xp9++);
            *(yp + 10) = *(xp10++);
            *(yp + 11) = *(xp11++);
            *(yp + 12) = *(xp12++);
            *(yp + 13) = *(xp13++);
            *(yp + 14) = *(xp14++);
            *(yp + 15) = *(xp15++);
            yp += ldY; 
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const float *xp0 = &X[index2(r, 0, ldX)];
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}

void pack_matrix_col2row_d(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const double *xp0 = &X[index2(c, 0, ldX)];
        const double *xp1 = &X[index2(c + 1, 0, ldX)];
        const double *xp2 = &X[index2(c + 2, 0, ldX)];
        const double *xp3 = &X[index2(c + 3, 0, ldX)];
        double *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = colX4 < 0 ? 0 : colX4; c < colX; c += 1)
    {
        const double *xp0 = &X[index2(c, 0, ldX)];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col_d(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = 8 > alpha_get_thread_num() ? alpha_get_thread_num() : 8;
    ALPHA_INT rowX4 = rowX - 7;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < rowX4; r += 8)
    {
        const double *xp0 = &X[index2(r, 0, ldX)];
        const double *xp1 = &X[index2(r + 1, 0, ldX)];
        const double *xp2 = &X[index2(r + 2, 0, ldX)];
        const double *xp3 = &X[index2(r + 3, 0, ldX)];
        const double *xp4 = &X[index2(r + 4, 0, ldX)];
        const double *xp5 = &X[index2(r + 5, 0, ldX)];
        const double *xp6 = &X[index2(r + 6, 0, ldX)];
        const double *xp7 = &X[index2(r + 7, 0, ldX)];
        // const double *xp8 = &X[index2(r + 8, 0, ldX)];
        // const double *xp9 = &X[index2(r + 9, 0, ldX)];
        // const double *xp10 = &X[index2(r + 10, 0, ldX)];
        // const double *xp11 = &X[index2(r + 11, 0, ldX)];
        // const double *xp12 = &X[index2(r + 12, 0, ldX)];
        // const double *xp13 = &X[index2(r + 13, 0, ldX)];
        // const double *xp14 = &X[index2(r + 14, 0, ldX)];
        // const double *xp15 = &X[index2(r + 15, 0, ldX)];
        double *yp = &Y[r];
        double *yp1 = yp + ldY;
        double *yp2 = yp1 + ldY;
        double *yp3 = yp2 + ldY;
        ALPHA_INT colX4 = colX - 3;
        ALPHA_INT c = 0;
        for (; c < colX4; c+=4)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            // *(yp + 8) = *(xp8++);
            // *(yp + 9) = *(xp9++);
            // *(yp + 10) = *(xp10++);
            // *(yp + 11) = *(xp11++);
            // *(yp + 12) = *(xp12++);
            // *(yp + 13) = *(xp13++);
            // *(yp + 14) = *(xp14++);
            // *(yp + 15) = *(xp15++);
            yp += 4*ldY;
            *yp1 = *(xp0++);
            *(yp1 + 1) = *(xp1++);
            *(yp1 + 2) = *(xp2++);
            *(yp1 + 3) = *(xp3++);
            *(yp1 + 4) = *(xp4++);
            *(yp1 + 5) = *(xp5++);
            *(yp1 + 6) = *(xp6++);
            *(yp1 + 7) = *(xp7++);
            // *(yp1 + 8) = *(xp8++);
            // *(yp1 + 9) = *(xp9++);
            // *(yp1 + 10) = *(xp10++);
            // *(yp1 + 11) = *(xp11++);
            // *(yp1 + 12) = *(xp12++);
            // *(yp1 + 13) = *(xp13++);
            // *(yp1 + 14) = *(xp14++);
            // *(yp1 + 15) = *(xp15++);
            yp1 += 4*ldY;

            *yp2 = *(xp0++);
            *(yp2 + 1) = *(xp1++);
            *(yp2 + 2) = *(xp2++);
            *(yp2 + 3) = *(xp3++);
            *(yp2 + 4) = *(xp4++);
            *(yp2 + 5) = *(xp5++);
            *(yp2 + 6) = *(xp6++);
            *(yp2 + 7) = *(xp7++);
            // *(yp2 + 8) = *(xp8++);
            // *(yp2 + 9) = *(xp9++);
            // *(yp2 + 10) = *(xp10++);
            // *(yp2 + 11) = *(xp11++);
            // *(yp2 + 12) = *(xp12++);
            // *(yp2 + 13) = *(xp13++);
            // *(yp2 + 14) = *(xp14++);
            // *(yp2 + 15) = *(xp15++);
            yp2 += 4*ldY;

            *yp3 = *(xp0++);
            *(yp3 + 1) = *(xp1++);
            *(yp3 + 2) = *(xp2++);
            *(yp3 + 3) = *(xp3++);
            *(yp3 + 4) = *(xp4++);
            *(yp3 + 5) = *(xp5++);
            *(yp3 + 6) = *(xp6++);
            *(yp3 + 7) = *(xp7++);
            // *(yp3 + 8) = *(xp8++);
            // *(yp3 + 9) = *(xp9++);
            // *(yp3 + 10) = *(xp10++);
            // *(yp3 + 11) = *(xp11++);
            // *(yp3 + 12) = *(xp12++);
            // *(yp3 + 13) = *(xp13++);
            // *(yp3 + 14) = *(xp14++);
            // *(yp3 + 15) = *(xp15++);
            yp3 += 4*ldY;
        }
        for (; c < colX; c++)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            // *(yp + 8) = *(xp8++);
            // *(yp + 9) = *(xp9++);
            // *(yp + 10) = *(xp10++);
            // *(yp + 11) = *(xp11++);
            // *(yp + 12) = *(xp12++);
            // *(yp + 13) = *(xp13++);
            // *(yp + 14) = *(xp14++);
            // *(yp + 15) = *(xp15++);
            yp += ldY; 
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const double *xp0 = &X[index2(r, 0, ldX)];
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}

void pack_matrix_col2row_c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(c, 0, ldX)];
        const ALPHA_Complex8 *xp1 = &X[index2(c + 1, 0, ldX)];
        const ALPHA_Complex8 *xp2 = &X[index2(c + 2, 0, ldX)];
        const ALPHA_Complex8 *xp3 = &X[index2(c + 3, 0, ldX)];
        ALPHA_Complex8 *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = (colX4 < 0 ? 0 : colX4); c < colX; c += 1)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(c, 0, ldX)];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col_c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT rowX4 = rowX - 3;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
    for (ALPHA_INT r = 0; r < rowX4; r += 4)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(r, 0, ldX)];
        const ALPHA_Complex8 *xp1 = &X[index2(r + 1, 0, ldX)];
        const ALPHA_Complex8 *xp2 = &X[index2(r + 2, 0, ldX)];
        const ALPHA_Complex8 *xp3 = &X[index2(r + 3, 0, ldX)];
        ALPHA_Complex8 *yp = &Y[r];
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(r, 0, ldX)];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}

void pack_matrix_col2row_z(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(c, 0, ldX)];
        const ALPHA_Complex16 *xp1 = &X[index2(c + 1, 0, ldX)];
        const ALPHA_Complex16 *xp2 = &X[index2(c + 2, 0, ldX)];
        const ALPHA_Complex16 *xp3 = &X[index2(c + 3, 0, ldX)];
        ALPHA_Complex16 *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = colX4 < 0 ? 0 : colX4; c < colX; c += 1)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(c, 0, ldX)];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col_z(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 * Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT rowX4 = rowX - 3;
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
    for (ALPHA_INT r = 0; r < rowX4; r += 4)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(r, 0, ldX)];
        const ALPHA_Complex16 *xp1 = &X[index2(r + 1, 0, ldX)];
        const ALPHA_Complex16 *xp2 = &X[index2(r + 2, 0, ldX)];
        const ALPHA_Complex16 *xp3 = &X[index2(r + 3, 0, ldX)];
        ALPHA_Complex16 *yp = &Y[r];
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(r, 0, ldX)];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}
