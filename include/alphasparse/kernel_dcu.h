#pragma once

#include "compute.h"
#include "format.h"

#include "kernel_dcu/kernel_x_dcu.h"

#include "kernel_dcu/kernel_s.h"
#include "kernel_dcu/kernel_d.h"
#include "kernel_dcu/kernel_c.h"
#include "kernel_dcu/kernel_z.h"

#include "kernel_dcu/kernel_csr_s_dcu.h"
#include "kernel_dcu/kernel_csr_d_dcu.h"
#include "kernel_dcu/kernel_csr_c_dcu.h"
#include "kernel_dcu/kernel_csr_z_dcu.h"

#include "kernel_dcu/kernel_coo_s_dcu.h"
#include "kernel_dcu/kernel_coo_d_dcu.h"
#include "kernel_dcu/kernel_coo_c_dcu.h"
#include "kernel_dcu/kernel_coo_z_dcu.h"

#include "kernel_dcu/kernel_bsr_s_dcu.h"
#include "kernel_dcu/kernel_bsr_d_dcu.h"
#include "kernel_dcu/kernel_bsr_c_dcu.h"
#include "kernel_dcu/kernel_bsr_z_dcu.h"

#include "kernel_dcu/kernel_gebsr_s_dcu.h"
#include "kernel_dcu/kernel_gebsr_d_dcu.h"
#include "kernel_dcu/kernel_gebsr_c_dcu.h"
#include "kernel_dcu/kernel_gebsr_z_dcu.h"

#include "kernel_dcu/kernel_ell_s_dcu.h"
#include "kernel_dcu/kernel_ell_d_dcu.h"
#include "kernel_dcu/kernel_ell_c_dcu.h"
#include "kernel_dcu/kernel_ell_z_dcu.h"

#include "kernel_dcu/kernel_hyb_s_dcu.h"
#include "kernel_dcu/kernel_hyb_d_dcu.h"
#include "kernel_dcu/kernel_hyb_c_dcu.h"
#include "kernel_dcu/kernel_hyb_z_dcu.h"

#include "kernel_dcu/kernel_other_s_dcu.h"
#include "kernel_dcu/kernel_other_d_dcu.h"
#include "kernel_dcu/kernel_other_c_dcu.h"
#include "kernel_dcu/kernel_other_z_dcu.h"

#ifndef COMPLEX
#ifndef DOUBLE
#include "kernel_dcu/def_s_dcu.h"
#else
#include "kernel_dcu/def_d_dcu.h"
#endif
#else
#ifndef DOUBLE
#include "kernel_dcu/def_c_dcu.h"
#else
#include "kernel_dcu/def_z_dcu.h"
#endif
#endif
