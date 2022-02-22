#pragma once

/**
 * @brief header for coo matrix related private interfaces
 */

#include "../spmat.h"
#include "../types.h"

alphasparse_status_t coo_s_order(spmat_coo_s_t *mat);
alphasparse_status_t coo_d_order(spmat_coo_d_t *mat);
alphasparse_status_t coo_c_order(spmat_coo_c_t *mat);
alphasparse_status_t coo_z_order(spmat_coo_z_t *mat);

alphasparse_status_t destroy_s_coo(spmat_coo_s_t *A);
alphasparse_status_t transpose_s_coo(const spmat_coo_s_t *s, spmat_coo_s_t **d);
alphasparse_status_t convert_csr_s_coo(const spmat_coo_s_t *source, spmat_csr_s_t **dest);
alphasparse_status_t convert_csc_s_coo(const spmat_coo_s_t *source, spmat_csc_s_t **dest);
alphasparse_status_t convert_bsr_s_coo(const spmat_coo_s_t *source, spmat_bsr_s_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_sky_s_coo(const spmat_coo_s_t *source, spmat_sky_s_t **dest,
                                      const alphasparse_fill_mode_t fill);
alphasparse_status_t convert_dia_s_coo(const spmat_coo_s_t *source, spmat_dia_s_t **dest);
alphasparse_status_t convert_ell_s_coo(const spmat_coo_s_t *source, spmat_ell_s_t **dest);
alphasparse_status_t convert_hints_ell_s_coo(const spmat_coo_s_t *source, spmat_ell_s_t **dest);
alphasparse_status_t convert_hints_dia_s_coo(const spmat_coo_s_t *source, spmat_dia_s_t **dest);
alphasparse_status_t convert_hints_bsr_s_coo(const spmat_coo_s_t *source, spmat_bsr_s_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_gebsr_s_coo(const spmat_coo_s_t *source, spmat_gebsr_s_t **dest,
                                        const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim,
                                        const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hyb_s_coo(const spmat_coo_s_t *source, spmat_hyb_s_t **dest);

alphasparse_status_t destroy_d_coo(spmat_coo_d_t *A);
alphasparse_status_t transpose_d_coo(const spmat_coo_d_t *s, spmat_coo_d_t **d);
alphasparse_status_t convert_csr_d_coo(const spmat_coo_d_t *source, spmat_csr_d_t **dest);
alphasparse_status_t convert_csc_d_coo(const spmat_coo_d_t *source, spmat_csc_d_t **dest);
alphasparse_status_t convert_bsr_d_coo(const spmat_coo_d_t *source, spmat_bsr_d_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_sky_d_coo(const spmat_coo_d_t *source, spmat_sky_d_t **dest,
                                      const alphasparse_fill_mode_t fill);
alphasparse_status_t convert_dia_d_coo(const spmat_coo_d_t *source, spmat_dia_d_t **dest);
alphasparse_status_t convert_ell_d_coo(const spmat_coo_d_t *source, spmat_ell_d_t **dest);
alphasparse_status_t convert_hints_ell_d_coo(const spmat_coo_d_t *source, spmat_ell_d_t **dest);
alphasparse_status_t convert_hints_dia_d_coo(const spmat_coo_d_t *source, spmat_dia_d_t **dest);
alphasparse_status_t convert_hints_bsr_d_coo(const spmat_coo_d_t *source, spmat_bsr_d_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_gebsr_d_coo(const spmat_coo_d_t *source, spmat_gebsr_d_t **dest,
                                        const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim,
                                        const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hyb_d_coo(const spmat_coo_d_t *source, spmat_hyb_d_t **dest);

alphasparse_status_t destroy_c_coo(spmat_coo_c_t *A);
alphasparse_status_t transpose_c_coo(const spmat_coo_c_t *s, spmat_coo_c_t **d);
alphasparse_status_t transpose_conj_c_coo(const spmat_coo_c_t *s, spmat_coo_c_t **d);
alphasparse_status_t convert_csr_c_coo(const spmat_coo_c_t *source, spmat_csr_c_t **dest);
alphasparse_status_t convert_csc_c_coo(const spmat_coo_c_t *source, spmat_csc_c_t **dest);
alphasparse_status_t convert_bsr_c_coo(const spmat_coo_c_t *source, spmat_bsr_c_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_sky_c_coo(const spmat_coo_c_t *source, spmat_sky_c_t **dest,
                                      const alphasparse_fill_mode_t fill);
alphasparse_status_t convert_dia_c_coo(const spmat_coo_c_t *source, spmat_dia_c_t **dest);
alphasparse_status_t convert_ell_c_coo(const spmat_coo_c_t *source, spmat_ell_c_t **dest);
alphasparse_status_t convert_hints_ell_c_coo(const spmat_coo_c_t *source, spmat_ell_c_t **dest);
alphasparse_status_t convert_hints_dia_c_coo(const spmat_coo_c_t *source, spmat_dia_c_t **dest);
alphasparse_status_t convert_hints_bsr_c_coo(const spmat_coo_c_t *source, spmat_bsr_c_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_gebsr_c_coo(const spmat_coo_c_t *source, spmat_gebsr_c_t **dest,
                                        const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim,
                                        const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hyb_c_coo(const spmat_coo_c_t *source, spmat_hyb_c_t **dest);

alphasparse_status_t destroy_z_coo(spmat_coo_z_t *A);
alphasparse_status_t transpose_z_coo(const spmat_coo_z_t *s, spmat_coo_z_t **d);
alphasparse_status_t transpose_conj_z_coo(const spmat_coo_z_t *s, spmat_coo_z_t **d);
alphasparse_status_t convert_csr_z_coo(const spmat_coo_z_t *source, spmat_csr_z_t **dest);
alphasparse_status_t convert_csc_z_coo(const spmat_coo_z_t *source, spmat_csc_z_t **dest);
alphasparse_status_t convert_bsr_z_coo(const spmat_coo_z_t *source, spmat_bsr_z_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_sky_z_coo(const spmat_coo_z_t *source, spmat_sky_z_t **dest,
                                      const alphasparse_fill_mode_t fill);
alphasparse_status_t convert_dia_z_coo(const spmat_coo_z_t *source, spmat_dia_z_t **dest);
alphasparse_status_t convert_ell_z_coo(const spmat_coo_z_t *source, spmat_ell_z_t **dest);
alphasparse_status_t convert_hints_ell_z_coo(const spmat_coo_z_t *source, spmat_ell_z_t **dest);
alphasparse_status_t convert_hints_dia_z_coo(const spmat_coo_z_t *source, spmat_dia_z_t **dest);
alphasparse_status_t convert_hints_bsr_z_coo(const spmat_coo_z_t *source, spmat_bsr_z_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_gebsr_z_coo(const spmat_coo_z_t *source, spmat_gebsr_z_t **dest,
                                        const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim,
                                        const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hyb_z_coo(const spmat_coo_z_t *source, spmat_hyb_z_t **dest);

alphasparse_status_t create_gen_from_special_s_coo(const spmat_coo_s_t *source, spmat_coo_s_t **dest,
                                                  struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_d_coo(const spmat_coo_d_t *source, spmat_coo_d_t **dest,
                                                  struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_c_coo(const spmat_coo_c_t *source, spmat_coo_c_t **dest,
                                                  struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_z_coo(const spmat_coo_z_t *source, spmat_coo_z_t **dest,
                                                  struct alpha_matrix_descr descr_in);