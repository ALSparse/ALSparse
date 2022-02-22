#pragma once

/**
 * @brief header for hyb matrix related private interfaces
 */

#include "../types.h"
#include "../spmat.h"

alphasparse_status_t destroy_s_hyb(spmat_hyb_s_t *A);
alphasparse_status_t transpose_s_hyb(const spmat_hyb_s_t *s, spmat_hyb_s_t **d);
alphasparse_status_t convert_coo_s_hyb(const spmat_hyb_s_t *source, spmat_coo_s_t **dest);
alphasparse_status_t convert_csr_s_hyb(const spmat_hyb_s_t *source, spmat_csr_s_t **dest);
alphasparse_status_t convert_csc_s_hyb(const spmat_hyb_s_t *source, spmat_csc_s_t **dest);
alphasparse_status_t convert_hyb_s_hyb(const spmat_hyb_s_t *source, spmat_hyb_s_t **dest, const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim, const alphasparse_layout_t block_layout);

alphasparse_status_t destroy_d_hyb(spmat_hyb_d_t *A);
alphasparse_status_t transpose_d_hyb(const spmat_hyb_d_t *s, spmat_hyb_d_t **d);
alphasparse_status_t convert_coo_d_hyb(const spmat_hyb_d_t *source, spmat_coo_d_t **dest);
alphasparse_status_t convert_csr_d_hyb(const spmat_hyb_d_t *source, spmat_csr_d_t **dest);
alphasparse_status_t convert_csc_d_hyb(const spmat_hyb_d_t *source, spmat_csc_d_t **dest);
alphasparse_status_t convert_hyb_d_hyb(const spmat_hyb_d_t *source, spmat_hyb_d_t **dest, const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim, const alphasparse_layout_t block_layout);


alphasparse_status_t destroy_c_hyb(spmat_hyb_c_t *A);
alphasparse_status_t transpose_c_hyb(const spmat_hyb_c_t *s, spmat_hyb_c_t **d);
alphasparse_status_t transpose_conj_c_hyb(const spmat_hyb_c_t *s, spmat_hyb_c_t **d);
alphasparse_status_t convert_coo_c_hyb(const spmat_hyb_c_t *source, spmat_coo_c_t **dest);
alphasparse_status_t convert_csr_c_hyb(const spmat_hyb_c_t *source, spmat_csr_c_t **dest);
alphasparse_status_t convert_csc_c_hyb(const spmat_hyb_c_t *source, spmat_csc_c_t **dest);
alphasparse_status_t convert_hyb_c_hyb(const spmat_hyb_c_t *source, spmat_hyb_c_t **dest, const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim, const alphasparse_layout_t block_layout);


alphasparse_status_t destroy_z_hyb(spmat_hyb_z_t *A);
alphasparse_status_t transpose_z_hyb(const spmat_hyb_z_t *s, spmat_hyb_z_t **d);
alphasparse_status_t transpose_conj_z_hyb(const spmat_hyb_z_t *s, spmat_hyb_z_t **d);
alphasparse_status_t convert_coo_z_hyb(const spmat_hyb_z_t *source, spmat_coo_z_t **dest);
alphasparse_status_t convert_csr_z_hyb(const spmat_hyb_z_t *source, spmat_csr_z_t **dest);
alphasparse_status_t convert_csc_z_hyb(const spmat_hyb_z_t *source, spmat_csc_z_t **dest);
alphasparse_status_t convert_hyb_z_hyb(const spmat_hyb_z_t *source, spmat_hyb_z_t **dest, const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim, const alphasparse_layout_t block_layout);
