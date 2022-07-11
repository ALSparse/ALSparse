#pragma once

/**
 * @brief header for all openspblas spblas public APIs;
 */ 

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include "alphasparse_cpu.h"  
#ifdef __DCU__
#include "alphasparse_dcu.h"
#include "alphasparse/spapi_dcu.h"  // spblas API for DCU
#endif

#include "alphasparse/spapi.h"
#include "alphasparse/spapi_uni.h"
#include "alphasparse/spapi_plain.h"  // spblas plain API

#ifdef __cplusplus
}
#endif /*__cplusplus */