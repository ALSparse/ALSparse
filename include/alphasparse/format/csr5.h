#pragma once

/**
 * @brief header for csr5 matrix related private interfaces
 */

#include "../spmat.h"
#include "../types.h"

alphasparse_status_t destroy_s_csr5(spmat_csr5_s_t *A);
alphasparse_status_t convert_csr_s_csr5(const spmat_csr5_s_t *source, spmat_csr_s_t **dest);

alphasparse_status_t destroy_d_csr5(spmat_csr5_d_t *A);
alphasparse_status_t convert_csr_d_csr5(const spmat_csr5_d_t *source, spmat_csr_d_t **dest);

alphasparse_status_t destroy_c_csr5(spmat_csr5_c_t *A);
alphasparse_status_t convert_csr_c_csr5(const spmat_csr5_c_t *source, spmat_csr_c_t **dest);

alphasparse_status_t destroy_z_csr5(spmat_csr5_z_t *A);
alphasparse_status_t convert_csr_z_csr5(const spmat_csr5_z_t *source, spmat_csr_z_t **dest);