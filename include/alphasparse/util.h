#pragma once

/**
 * @brief header for all utils
 */

#include "util/assert.h"
#include "util/check.h"
#include "util/error.h"
#include "util/malloc.h"
#include "util/timing.h"
#include "util/partition.h"
#include "util/args.h"
#include "util/pack.h"
#include "util/bisearch.h"
#include "util/analysis.h"
#include "util/norm.h"

#include "util/vector_fma2.h"
#include "util/vector_doti.h"
#include "util/vector_blkmm.h"
#include "util/vector_blkmv.h"

#include "util/io.h"
#include "util/algebra.h"

#ifndef index2
#define index2(y, x, ldx) ((x) + (ldx) * (y))
#endif // !index2

#ifndef index3
#define index3(z, y, x, ldy, ldx) index2(index2(z, y, ldy), x, ldx)
#endif // !index3

#ifndef index4
#define index4(d, c, b, a, ldc, ldb, lda) index2(index2(index2(d, c, ldc), b, ldb), a, lda)
#endif // !index4

#ifndef alpha_min
#define alpha_min(x, y) ((x) < (y) ? (x) : (y))
#endif // !alpha_min

#ifndef alpha_max
#define alpha_max(x, y) ((x) < (y) ? (y) : (x))
#endif // !alpha_max
