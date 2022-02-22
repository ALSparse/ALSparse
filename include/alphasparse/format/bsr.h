#pragma once

/**
 * @brief header for bsr matrix related private interfaces
 */

#include "../types.h"
#include "../spmat.h"

alphasparse_status_t destroy_s_bsr(spmat_bsr_s_t *A);
alphasparse_status_t transpose_s_bsr(const spmat_bsr_s_t *s, spmat_bsr_s_t **d);
alphasparse_status_t convert_coo_s_bsr(const spmat_bsr_s_t *source, spmat_coo_s_t **dest);
alphasparse_status_t convert_csr_s_bsr(const spmat_bsr_s_t *source, spmat_csr_s_t **dest);
alphasparse_status_t convert_csc_s_bsr(const spmat_bsr_s_t *source, spmat_csc_s_t **dest);
alphasparse_status_t convert_bsr_s_bsr(const spmat_bsr_s_t *source, spmat_bsr_s_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);

alphasparse_status_t destroy_d_bsr(spmat_bsr_d_t *A);
alphasparse_status_t transpose_d_bsr(const spmat_bsr_d_t *s, spmat_bsr_d_t **d);
alphasparse_status_t convert_coo_d_bsr(const spmat_bsr_d_t *source, spmat_coo_d_t **dest);
alphasparse_status_t convert_csr_d_bsr(const spmat_bsr_d_t *source, spmat_csr_d_t **dest);
alphasparse_status_t convert_csc_d_bsr(const spmat_bsr_d_t *source, spmat_csc_d_t **dest);
alphasparse_status_t convert_bsr_d_bsr(const spmat_bsr_d_t *source, spmat_bsr_d_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);


alphasparse_status_t destroy_c_bsr(spmat_bsr_c_t *A);
alphasparse_status_t transpose_c_bsr(const spmat_bsr_c_t *s, spmat_bsr_c_t **d);
alphasparse_status_t transpose_conj_c_bsr(const spmat_bsr_c_t *s, spmat_bsr_c_t **d);
alphasparse_status_t convert_coo_c_bsr(const spmat_bsr_c_t *source, spmat_coo_c_t **dest);
alphasparse_status_t convert_csr_c_bsr(const spmat_bsr_c_t *source, spmat_csr_c_t **dest);
alphasparse_status_t convert_csc_c_bsr(const spmat_bsr_c_t *source, spmat_csc_c_t **dest);
alphasparse_status_t convert_bsr_c_bsr(const spmat_bsr_c_t *source, spmat_bsr_c_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);


alphasparse_status_t destroy_z_bsr(spmat_bsr_z_t *A);
alphasparse_status_t transpose_z_bsr(const spmat_bsr_z_t *s, spmat_bsr_z_t **d);
alphasparse_status_t transpose_conj_z_bsr(const spmat_bsr_z_t *s, spmat_bsr_z_t **d);
alphasparse_status_t convert_coo_z_bsr(const spmat_bsr_z_t *source, spmat_coo_z_t **dest);
alphasparse_status_t convert_csr_z_bsr(const spmat_bsr_z_t *source, spmat_csr_z_t **dest);
alphasparse_status_t convert_csc_z_bsr(const spmat_bsr_z_t *source, spmat_csc_z_t **dest);
alphasparse_status_t convert_bsr_z_bsr(const spmat_bsr_z_t *source, spmat_bsr_z_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);
