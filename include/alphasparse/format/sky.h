#pragma once

/**
 * @brief header for sky matrix related private interfaces
 */

#include "../types.h"
#include "../spmat.h"

alphasparse_status_t destroy_s_sky(spmat_sky_s_t *A);
alphasparse_status_t transpose_s_sky(const spmat_sky_s_t *s, spmat_sky_s_t **d);

alphasparse_status_t destroy_d_sky(spmat_sky_d_t *A);
alphasparse_status_t transpose_d_sky(const spmat_sky_d_t *s, spmat_sky_d_t **d);

alphasparse_status_t destroy_c_sky(spmat_sky_c_t *A);
alphasparse_status_t transpose_c_sky(const spmat_sky_c_t *s, spmat_sky_c_t **d);
alphasparse_status_t transpose_conj_c_sky(const spmat_sky_c_t *s, spmat_sky_c_t **d);

alphasparse_status_t destroy_z_sky(spmat_sky_z_t *A);
alphasparse_status_t transpose_z_sky(const spmat_sky_z_t *s, spmat_sky_z_t **d);
alphasparse_status_t transpose_conj_z_sky(const spmat_sky_z_t *s, spmat_sky_z_t **d);
