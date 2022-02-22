#pragma once

/**
 * @brief header for csr matrix related private interfaces
 */

#include "../spmat.h"
#include "../types.h"
alphasparse_status_t csr_s_order(spmat_csr_s_t *mat);
alphasparse_status_t csr_d_order(spmat_csr_d_t *mat);
alphasparse_status_t csr_c_order(spmat_csr_c_t *mat);
alphasparse_status_t csr_z_order(spmat_csr_z_t *mat);

alphasparse_status_t destroy_s_csr(spmat_csr_s_t *A);
alphasparse_status_t transpose_s_csr(const spmat_csr_s_t *s, spmat_csr_s_t **d);
alphasparse_status_t convert_coo_s_csr(const spmat_csr_s_t *source, spmat_coo_s_t **dest);
alphasparse_status_t convert_csr_s_csr(const spmat_csr_s_t *source, spmat_csr_s_t **dest);
alphasparse_status_t convert_csr5_s_csr(const spmat_csr_s_t *source, spmat_csr5_s_t **dest);
alphasparse_status_t convert_csc_s_csr(const spmat_csr_s_t *source, spmat_csc_s_t **dest);
alphasparse_status_t convert_bsr_s_csr(const spmat_csr_s_t *source, spmat_bsr_s_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_dia_s_csr(const spmat_csr_s_t *source, spmat_dia_s_t **dest);
alphasparse_status_t convert_hints_bsr_s_csr(const spmat_csr_s_t *source, spmat_bsr_s_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hints_dia_s_csr(const spmat_csr_s_t *source, spmat_dia_s_t **dest);
alphasparse_status_t convert_hints_ell_s_csr(const spmat_csr_s_t *source, spmat_ell_s_t **dest);

alphasparse_status_t destroy_d_csr(spmat_csr_d_t *A);
alphasparse_status_t transpose_d_csr(const spmat_csr_d_t *s, spmat_csr_d_t **d);
alphasparse_status_t convert_coo_d_csr(const spmat_csr_d_t *source, spmat_coo_d_t **dest);
alphasparse_status_t convert_csr_d_csr(const spmat_csr_d_t *source, spmat_csr_d_t **dest);
alphasparse_status_t convert_csr5_d_csr(const spmat_csr_d_t *source, spmat_csr5_d_t **dest);
alphasparse_status_t convert_csc_d_csr(const spmat_csr_d_t *source, spmat_csc_d_t **dest);
alphasparse_status_t convert_bsr_d_csr(const spmat_csr_d_t *source, spmat_bsr_d_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_dia_d_csr(const spmat_csr_d_t *source, spmat_dia_d_t **dest);
alphasparse_status_t convert_hints_bsr_d_csr(const spmat_csr_d_t *source, spmat_bsr_d_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hints_dia_d_csr(const spmat_csr_d_t *source, spmat_dia_d_t **dest);
alphasparse_status_t convert_hints_ell_d_csr(const spmat_csr_d_t *source, spmat_ell_d_t **dest);

alphasparse_status_t destroy_c_csr(spmat_csr_c_t *A);
alphasparse_status_t transpose_c_csr(const spmat_csr_c_t *s, spmat_csr_c_t **d);
alphasparse_status_t transpose_conj_c_csr(const spmat_csr_c_t *s, spmat_csr_c_t **d);
alphasparse_status_t convert_coo_c_csr(const spmat_csr_c_t *source, spmat_coo_c_t **dest);
alphasparse_status_t convert_csr_c_csr(const spmat_csr_c_t *source, spmat_csr_c_t **dest);
alphasparse_status_t convert_csr5_c_csr(const spmat_csr_c_t *source, spmat_csr5_c_t **dest);
alphasparse_status_t convert_csc_c_csr(const spmat_csr_c_t *source, spmat_csc_c_t **dest);
alphasparse_status_t convert_bsr_c_csr(const spmat_csr_c_t *source, spmat_bsr_c_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_dia_c_csr(const spmat_csr_c_t *source, spmat_dia_c_t **dest);
alphasparse_status_t convert_hints_bsr_c_csr(const spmat_csr_c_t *source, spmat_bsr_c_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hints_dia_c_csr(const spmat_csr_c_t *source, spmat_dia_c_t **dest);
alphasparse_status_t convert_hints_ell_c_csr(const spmat_csr_c_t *source, spmat_ell_c_t **dest);

alphasparse_status_t destroy_z_csr(spmat_csr_z_t *A);
alphasparse_status_t transpose_z_csr(const spmat_csr_z_t *s, spmat_csr_z_t **d);
alphasparse_status_t transpose_conj_z_csr(const spmat_csr_z_t *s, spmat_csr_z_t **d);
alphasparse_status_t convert_coo_z_csr(const spmat_csr_z_t *source, spmat_coo_z_t **dest);
alphasparse_status_t convert_csr_z_csr(const spmat_csr_z_t *source, spmat_csr_z_t **dest);
alphasparse_status_t convert_csr5_z_csr(const spmat_csr_z_t *source, spmat_csr5_z_t **dest);
alphasparse_status_t convert_csc_z_csr(const spmat_csr_z_t *source, spmat_csc_z_t **dest);
alphasparse_status_t convert_bsr_z_csr(const spmat_csr_z_t *source, spmat_bsr_z_t **dest,
                                      const ALPHA_INT block_size,
                                      const alphasparse_layout_t block_layout);
alphasparse_status_t convert_dia_z_csr(const spmat_csr_z_t *source, spmat_dia_z_t **dest);
alphasparse_status_t convert_hints_bsr_z_csr(const spmat_csr_z_t *source, spmat_bsr_z_t **dest,
                                            const ALPHA_INT block_size,
                                            const alphasparse_layout_t block_layout);
alphasparse_status_t convert_hints_dia_z_csr(const spmat_csr_z_t *source, spmat_dia_z_t **dest);
alphasparse_status_t convert_hints_ell_z_csr(const spmat_csr_z_t *source, spmat_ell_z_t **dest);

alphasparse_status_t create_gen_from_special_s_csr(const spmat_csr_s_t *source, spmat_csr_s_t **dest,
                                                  struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_d_csr(const spmat_csr_d_t *source, spmat_csr_d_t **dest,
                                                  struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_c_csr(const spmat_csr_c_t *source, spmat_csr_c_t **dest,
                                                  struct alpha_matrix_descr descr_in);
alphasparse_status_t create_gen_from_special_z_csr(const spmat_csr_z_t *source, spmat_csr_z_t **dest,
                                                  struct alpha_matrix_descr descr_in);
