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
#endif

#include "alphasparse/spapi_uni.h"

#ifdef __cplusplus
}
#endif /*__cplusplus */