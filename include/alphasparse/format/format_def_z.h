#pragma once

#include "coo.h"
#include "csr.h"
#include "csc.h"
#include "bsr.h"
#include "dia.h"
#include "sky.h"
#include "ell.h"
#include "csr5.h"

#define destroy_coo destroy_z_coo
#define transpose_coo transpose_z_coo
#define transpose_conj_coo transpose_conj_z_coo
#define convert_csr_coo convert_csr_z_coo
#define convert_csc_coo convert_csc_z_coo
#define convert_bsr_coo convert_bsr_z_coo
#define convert_sky_coo convert_sky_z_coo
#define convert_dia_coo convert_dia_z_coo
#define convert_ell_coo convert_ell_z_coo
#define convert_hints_ell_coo convert_hints_ell_z_coo
#define convert_hints_ell_coo convert_hints_ell_z_coo
#define convert_hints_ell_coo convert_hints_ell_z_coo

#define destroy_csr destroy_z_csr
#define transpose_csr transpose_z_csr
#define transpose_conj_csr transpose_conj_z_csr
#define csr_order csr_z_order
#define bsr_order bsr_z_order
#define coo_order coo_z_order
#define convert_coo_csr convert_coo_z_csr
#define convert_csr_csr convert_csr_z_csr
#define convert_csc_csr convert_csc_z_csr
#define convert_bsr_csr convert_bsr_z_csr
#define convert_csr5_csr convert_csr5_z_csr

#define destroy_csc destroy_z_csc
#define transpose_csc transpose_z_csc
#define transpose_conj_csc transpose_conj_z_csc
#define convert_coo_csc convert_coo_z_csc
#define convert_csr_csc convert_csr_z_csc
#define convert_csc_csc convert_csc_z_csc
#define convert_bsr_csc convert_bsr_z_csc

#define destroy_bsr destroy_z_bsr
#define transpose_bsr transpose_z_bsr
#define transpose_conj_bsr transpose_conj_z_bsr
#define convert_coo_bsr convert_coo_z_bsr
#define convert_csr_bsr convert_csr_z_bsr
#define convert_csc_bsr convert_csc_z_bsr
#define convert_bsr_bsr convert_bsr_z_bsr

#define destroy_sky destroy_z_sky
#define transpose_sky transpose_z_sky
#define transpose_conj_sky transpose_conj_z_sky

#define destroy_dia destroy_z_dia
#define transpose_dia transpose_z_dia
#define transpose_conj_dia transpose_conj_z_dia

#define destroy_ell destroy_z_ell
#define transpose_ell transpose_z_ell
#define transpose_conj_ell transpose_conj_z_ell

#define destroy_hyb destroy_z_hyb
#define transpose_hyb transpose_z_hyb
#define transpose_conj_hyb transpose_conj_z_hyb

#define destroy_gebsr destroy_z_gebsr
#define transpose_gebsr transpose_z_gebsr
#define transpose_conj_gebsr transpose_conj_z_gebsr

#define destroy_csr5 destroy_z_csr5
#define convert_csr_csr5 convert_csr_z_csr5

#define create_gen_from_special_csr create_gen_from_special_z_csr
#define create_gen_from_special_bsr create_gen_from_special_z_bsr
#define create_gen_from_special_coo create_gen_from_special_z_coo
#define create_gen_from_special_csc create_gen_from_special_z_csc
#define create_gen_from_special_dia create_gen_from_special_z_dia
#define create_gen_from_special_ell create_gen_from_special_z_ell
