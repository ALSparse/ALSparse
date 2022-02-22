#pragma once

#include "coo.h"
#include "csr.h"
#include "csc.h"
#include "bsr.h"
#include "dia.h"
#include "sky.h"
#include "ell.h"
#include "csr5.h"

#define destroy_coo destroy_d_coo
#define transpose_coo transpose_d_coo
#define transpose_conj_coo transpose_conj_d_coo
#define convert_csr_coo convert_csr_d_coo
#define convert_csc_coo convert_csc_d_coo
#define convert_bsr_coo convert_bsr_d_coo
#define convert_sky_coo convert_sky_d_coo
#define convert_dia_coo convert_dia_d_coo
#define convert_ell_coo convert_ell_d_coo
#define convert_hints_ell_coo convert_hints_ell_d_coo
#define convert_hints_ell_coo convert_hints_ell_d_coo
#define convert_hints_ell_coo convert_hints_ell_d_coo

#define destroy_csr destroy_d_csr
#define transpose_csr transpose_d_csr
#define transpose_conj_csr transpose_conj_d_csr
#define csr_order csr_d_order
#define bsr_order bsr_d_order
#define coo_order coo_d_order
#define convert_coo_csr convert_coo_d_csr
#define convert_csr_csr convert_csr_d_csr
#define convert_csc_csr convert_csc_d_csr
#define convert_bsr_csr convert_bsr_d_csr
#define convert_csr5_csr convert_csr5_d_csr

#define destroy_csc destroy_d_csc
#define transpose_csc transpose_d_csc
#define transpose_conj_csc transpose_conj_d_csc
#define convert_coo_csc convert_coo_d_csc
#define convert_csr_csc convert_csr_d_csc
#define convert_csc_csc convert_csc_d_csc
#define convert_bsr_csc convert_bsr_d_csc

#define destroy_bsr destroy_d_bsr
#define transpose_bsr transpose_d_bsr
#define transpose_conj_bsr transpose_conj_d_bsr
#define convert_coo_bsr convert_coo_d_bsr
#define convert_csr_bsr convert_csr_d_bsr
#define convert_csc_bsr convert_csc_d_bsr
#define convert_bsr_bsr convert_bsr_d_bsr

#define destroy_sky destroy_d_sky
#define transpose_sky transpose_d_sky
#define transpose_conj_sky transpose_conj_d_sky

#define destroy_dia destroy_d_dia
#define transpose_dia transpose_d_dia
#define transpose_conj_dia transpose_conj_d_dia

#define destroy_ell destroy_d_ell
#define transpose_ell transpose_d_ell
#define transpose_conj_ell transpose_conj_d_ell

#define destroy_hyb destroy_d_hyb
#define transpose_hyb transpose_d_hyb
#define transpose_conj_hyb transpose_conj_d_hyb

#define destroy_gebsr destroy_d_gebsr
#define transpose_gebsr transpose_d_gebsr
#define transpose_conj_gebsr transpose_conj_d_gebsr

#define destroy_csr5 destroy_d_csr5
#define convert_csr_csr5 convert_csr_d_csr5

#define create_gen_from_special_csr create_gen_from_special_d_csr
#define create_gen_from_special_bsr create_gen_from_special_d_bsr
#define create_gen_from_special_coo create_gen_from_special_d_coo
#define create_gen_from_special_csc create_gen_from_special_d_csc
#define create_gen_from_special_dia create_gen_from_special_d_dia
#define create_gen_from_special_ell create_gen_from_special_d_ell
