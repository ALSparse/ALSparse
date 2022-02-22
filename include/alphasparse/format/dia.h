#pragma once

/**
 * @brief header for dia matrix related private interfaces
 */

#include "../types.h"
#include "../spmat.h"

alphasparse_status_t destroy_s_dia(spmat_dia_s_t *A);
alphasparse_status_t transpose_s_dia(const spmat_dia_s_t *s, spmat_dia_s_t **d);

alphasparse_status_t destroy_d_dia(spmat_dia_d_t *A);
alphasparse_status_t transpose_d_dia(const spmat_dia_d_t *s, spmat_dia_d_t **d);

alphasparse_status_t destroy_c_dia(spmat_dia_c_t *A);
alphasparse_status_t transpose_c_dia(const spmat_dia_c_t *s, spmat_dia_c_t **d);
alphasparse_status_t transpose_conj_c_dia(const spmat_dia_c_t *s, spmat_dia_c_t **d);

alphasparse_status_t destroy_z_dia(spmat_dia_z_t *A);
alphasparse_status_t transpose_z_dia(const spmat_dia_z_t *s, spmat_dia_z_t **d);
alphasparse_status_t transpose_conj_z_dia(const spmat_dia_z_t *s, spmat_dia_z_t **d);