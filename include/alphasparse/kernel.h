#pragma once

#include "compute.h"
#include "format.h"

#include "kernel/kernel_s.h"
#include "kernel/kernel_coo_s.h"
#include "kernel/kernel_csr_s.h"
#include "kernel/kernel_csc_s.h"
#include "kernel/kernel_bsr_s.h"
#include "kernel/kernel_sky_s.h"
#include "kernel/kernel_dia_s.h"
#include "kernel/kernel_d.h"
#include "kernel/kernel_coo_d.h"
#include "kernel/kernel_csr_d.h"
#include "kernel/kernel_csc_d.h"
#include "kernel/kernel_bsr_d.h"
#include "kernel/kernel_sky_d.h"
#include "kernel/kernel_dia_d.h"
#include "kernel/kernel_c.h"
#include "kernel/kernel_coo_c.h"
#include "kernel/kernel_csr_c.h"
#include "kernel/kernel_csc_c.h"
#include "kernel/kernel_bsr_c.h"
#include "kernel/kernel_sky_c.h"
#include "kernel/kernel_dia_c.h"
#include "kernel/kernel_z.h"
#include "kernel/kernel_coo_z.h"
#include "kernel/kernel_csr_z.h"
#include "kernel/kernel_csc_z.h"
#include "kernel/kernel_bsr_z.h"
#include "kernel/kernel_sky_z.h"
#include "kernel/kernel_dia_z.h"
#ifndef COMPLEX
#ifndef DOUBLE
#include "kernel/def_s.h"
#else
#include "kernel/def_d.h"
#endif
#else
#ifndef DOUBLE
#include "kernel/def_c.h"
#else
#include "kernel/def_z.h"
#endif
#endif
