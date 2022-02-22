#pragma once

#include "format/coo.h"
#include "format/csr.h"
#include "format/csc.h"
#include "format/bsr.h"
#include "format/sky.h"
#include "format/dia.h"
#include "format/ell.h"
#include "format/hyb.h"
#include "format/gebsr.h"

#ifndef COMPLEX
#ifndef DOUBLE
#include "format/format_def_s.h"
#else
#include "format/format_def_d.h"
#endif
#else
#ifndef DOUBLE
#include "format/format_def_c.h"
#else
#include "format/format_def_z.h"
#endif
#endif

