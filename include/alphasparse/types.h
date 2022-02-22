#pragma once

/**
 * @brief header for all basic type definitions except the internal sparse matrix
 */ 

#include <stdint.h>
#include <stdbool.h>

#ifndef DOUBLE
#define ALPHA_Float float
#define ALPHA_Complex ALPHA_Complex8
#else 
#define ALPHA_Float double
#define ALPHA_Complex ALPHA_Complex16
#endif

#ifndef COMPLEX
#define ALPHA_Number ALPHA_Float
#else 
#define ALPHA_Number ALPHA_Complex
#endif

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_Point point_s_t
#define S S
#else 
#define ALPHA_Point point_d_t
#define D D
#endif
#else
#ifndef DOUBLE
#define ALPHA_Point point_c_t
#define C C
#else 
#define ALPHA_Point point_z_t
#define Z Z
#endif
#endif
 

#ifndef ALPHA_Complex8
typedef
struct {
    float real;
    float imag;
} ALPHA_Complex8;
#endif

#ifndef ALPHA_Complex16
typedef
struct {
    double real;
    double imag;
} ALPHA_Complex16;
#endif

#ifndef ALPHA_INT
    #define ALPHA_INT int32_t
#endif

#ifndef ALPHA_UINT
    #define ALPHA_UINT uint32_t
#endif

#ifndef ALPHA_LONG
    #define ALPHA_LONG int64_t
#endif

#ifndef ALPHA_UINT8
    #define ALPHA_UINT8 uint8_t
#endif

#ifndef ALPHA_INT8
    #define ALPHA_INT8 int8_t
#endif

#ifndef ALPHA_INT16
    #define ALPHA_INT16 int16_t
#endif

#ifndef ALPHA_INT32
    #define ALPHA_INT32 int32_t
#endif

#ifndef ALPHA_INT64
    #define ALPHA_INT64 int64_t
#endif

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
} int2_t;

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
    ALPHA_INT z;
} int3_t;

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
    float v;
} point_s_t;

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
    double v;
} point_d_t;

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
    ALPHA_Complex8 v;
} point_c_t;

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
    ALPHA_Complex16 v;
} point_z_t;
