#pragma once

#include "alphasparse/handle.h"  // handle for DCU
#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include "alphasparse/types.h"  // basic type define
// #include "alphasparse/opt.h"    // optimazation
#include "alphasparse/spdef.h"  // spblas type define 
#include "alphasparse/spapi.h"  // spblas API
#include "alphasparse/spapi_dcu.h"  // spblas API for DCU

#include "alphasparse/spapi_plain.h"  // spblas plain API

#include "alphasparse/util/assert.h"
#include "alphasparse/util/error.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/util/timing.h"
#include "alphasparse/util/partition.h"
#include "alphasparse/util/bisearch.h"
#include "alphasparse/util/analysis.h"
#include "alphasparse/util/algebra.h"
#include "alphasparse/util/args.h"
#include "alphasparse/util/io.h"
#include "alphasparse/util/check.h"
#ifdef __cplusplus
}
#endif /*__cplusplus */