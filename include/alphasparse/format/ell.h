#pragma once

/**
 * @brief header for ell matrix related private interfaces
 */

#include "../types.h"
#include "../spmat.h"

alphasparse_status_t destroy_s_ell(spmat_ell_s_t *A);
alphasparse_status_t transpose_s_ell(const spmat_ell_s_t *s, spmat_ell_s_t **d);
alphasparse_status_t convert_coo_s_ell(const spmat_ell_s_t *source, spmat_coo_s_t **dest);
alphasparse_status_t convert_ell_s_ell(const spmat_ell_s_t *source, spmat_ell_s_t **dest);
alphasparse_status_t convert_csc_s_ell(const spmat_ell_s_t *source, spmat_csc_s_t **dest);
alphasparse_status_t convert_bsr_s_ell(const spmat_ell_s_t *source, spmat_bsr_s_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);

// alphasparse_status_t destroy_d_ell(spmat_ell_d_t *A);
// alphasparse_status_t transpose_d_ell(const spmat_ell_d_t *s, spmat_ell_d_t **d);
// alphasparse_status_t convert_coo_d_ell(const spmat_ell_d_t *source, spmat_coo_d_t **dest);
// alphasparse_status_t convert_ell_d_ell(const spmat_ell_d_t *source, spmat_ell_d_t **dest);
// alphasparse_status_t convert_csc_d_ell(const spmat_ell_d_t *source, spmat_csc_d_t **dest);
// alphasparse_status_t convert_bsr_d_ell(const spmat_ell_d_t *source, spmat_bsr_d_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);

// alphasparse_status_t destroy_c_ell(spmat_ell_c_t *A);
// alphasparse_status_t transpose_c_ell(const spmat_ell_c_t *s, spmat_ell_c_t **d);
// alphasparse_status_t transpose_conj_c_ell(const spmat_ell_c_t *s, spmat_ell_c_t **d);
// alphasparse_status_t convert_coo_c_ell(const spmat_ell_c_t *source, spmat_coo_c_t **dest);
// alphasparse_status_t convert_ell_c_ell(const spmat_ell_c_t *source, spmat_ell_c_t **dest);
// alphasparse_status_t convert_csc_c_ell(const spmat_ell_c_t *source, spmat_csc_c_t **dest);
// alphasparse_status_t convert_bsr_c_ell(const spmat_ell_c_t *source, spmat_bsr_c_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);

// alphasparse_status_t destroy_z_ell(spmat_ell_z_t *A);
// alphasparse_status_t transpose_z_ell(const spmat_ell_z_t *s, spmat_ell_z_t **d);
// alphasparse_status_t transpose_conj_z_ell(const spmat_ell_z_t *s, spmat_ell_z_t **d);
// alphasparse_status_t convert_coo_z_ell(const spmat_ell_z_t *source, spmat_coo_z_t **dest);
// alphasparse_status_t convert_ell_z_ell(const spmat_ell_z_t *source, spmat_ell_z_t **dest);
// alphasparse_status_t convert_csc_z_ell(const spmat_ell_z_t *source, spmat_csc_z_t **dest);
// alphasparse_status_t convert_bsr_z_ell(const spmat_ell_z_t *source, spmat_bsr_z_t **dest, const ALPHA_INT block_size, const alphasparse_layout_t block_layout);

// ELL needn't support trmv actually
alphasparse_status_t create_gen_from_special_s_ell(const spmat_ell_s_t *source, spmat_ell_s_t **dest, struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_d_ell(const spmat_ell_d_t *source, spmat_ell_d_t **dest, struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_c_ell(const spmat_ell_c_t *source, spmat_ell_c_t **dest, struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_z_ell(const spmat_ell_z_t *source, spmat_ell_z_t **dest, struct alpha_matrix_descr descr_in);