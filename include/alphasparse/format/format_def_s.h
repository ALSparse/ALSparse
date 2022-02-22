#pragma once

#include "coo.h"
#include "csr.h"
#include "csc.h"
#include "bsr.h"
#include "dia.h"
#include "sky.h"
#include "ell.h"
#include "csr5.h"

#define destroy_coo destroy_s_coo
#define transpose_coo transpose_s_coo
#define transpose_conj_coo transpose_conj_s_coo
#define convert_csr_coo convert_csr_s_coo
#define convert_csc_coo convert_csc_s_coo
#define convert_bsr_coo convert_bsr_s_coo
#define convert_sky_coo convert_sky_s_coo
#define convert_dia_coo convert_dia_s_coo
#define convert_ell_coo convert_ell_s_coo
#define convert_hints_ell_coo convert_hints_ell_s_coo
#define convert_hints_ell_coo convert_hints_ell_s_coo
#define convert_hints_ell_coo convert_hints_ell_s_coo

#define destroy_csr destroy_s_csr
#define transpose_csr transpose_s_csr
#define transpose_conj_csr transpose_conj_s_csr
#define csr_order csr_s_order
#define bsr_order bsr_s_order
#define coo_order coo_s_order
#define convert_coo_csr convert_coo_s_csr
#define convert_csr_csr convert_csr_s_csr
#define convert_csc_csr convert_csc_s_csr
#define convert_bsr_csr convert_bsr_s_csr
#define convert_csr5_csr convert_csr5_s_csr

#define destroy_csc destroy_s_csc
#define transpose_csc transpose_s_csc
#define transpose_conj_csc transpose_conj_s_csc
#define convert_coo_csc convert_coo_s_csc
#define convert_csr_csc convert_csr_s_csc
#define convert_csc_csc convert_csc_s_csc
#define convert_bsr_csc convert_bsr_s_csc

#define destroy_bsr destroy_s_bsr
#define transpose_bsr transpose_s_bsr
#define transpose_conj_bsr transpose_conj_s_bsr
#define convert_coo_bsr convert_coo_s_bsr
#define convert_csr_bsr convert_csr_s_bsr
#define convert_csc_bsr convert_csc_s_bsr
#define convert_bsr_bsr convert_bsr_s_bsr

#define destroy_sky destroy_s_sky
#define transpose_sky transpose_s_sky
#define transpose_conj_sky transpose_conj_s_sky

#define destroy_dia destroy_s_dia
#define transpose_dia transpose_s_dia
#define transpose_conj_dia transpose_conj_s_dia

#define destroy_ell destroy_s_ell
#define transpose_ell transpose_s_ell
#define transpose_conj_ell transpose_conj_s_ell

#define destroy_hyb destroy_s_hyb
#define transpose_hyb transpose_s_hyb
#define transpose_conj_hyb transpose_conj_s_hyb

#define destroy_gebsr destroy_s_gebsr
#define transpose_gebsr transpose_s_gebsr
#define transpose_conj_gebsr transpose_conj_s_gebsr

#define destroy_csr5 destroy_s_csr5
#define convert_csr_csr5 convert_csr_s_csr5

#define create_gen_from_special_csr create_gen_from_special_s_csr
#define create_gen_from_special_bsr create_gen_from_special_s_bsr
#define create_gen_from_special_coo create_gen_from_special_s_coo
#define create_gen_from_special_csc create_gen_from_special_s_csc
#define create_gen_from_special_dia create_gen_from_special_s_dia
// #define create_gen_from_special_ell create_gen_from_special_s_csr
#define create_gen_from_special_ell create_gen_from_special_s_ell
