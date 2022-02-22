#pragma once

alphasparse_status_t axpy_z_plain(const ALPHA_INT nz,  const ALPHA_Complex16 a,  const ALPHA_Complex16* x,  const ALPHA_INT* indx,  ALPHA_Complex16* y);
alphasparse_status_t gthr_z_plain(const ALPHA_INT nz, const ALPHA_Complex16* y, ALPHA_Complex16* x, const ALPHA_INT* indx);
alphasparse_status_t gthrz_z_plain(const ALPHA_INT nz, ALPHA_Complex16* y, ALPHA_Complex16* x, const ALPHA_INT* indx);
alphasparse_status_t sctr_z_plain(const ALPHA_INT nz, const ALPHA_Complex16* x, const ALPHA_INT* indx, ALPHA_Complex16* y);
void dotci_z_sub_plain(const ALPHA_INT nz,  const ALPHA_Complex16* x,  const ALPHA_INT* indx, const ALPHA_Complex16* y, ALPHA_Complex16 *dutci);
void dotui_z_sub_plain(const ALPHA_INT nz,  const ALPHA_Complex16* x,  const ALPHA_INT* indx, const ALPHA_Complex16* y, ALPHA_Complex16 *dutui);
