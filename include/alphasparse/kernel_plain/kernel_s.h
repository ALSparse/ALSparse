#pragma once

alphasparse_status_t axpy_s_plain(const ALPHA_INT nz,  const float a,  const float* x,  const ALPHA_INT* indx,  float* y);
alphasparse_status_t gthr_s_plain(const ALPHA_INT nz,	const float* y, float* x, const ALPHA_INT* indx);
alphasparse_status_t gthrz_s_plain(const ALPHA_INT nz, float* y, float* x, const ALPHA_INT* indx);
alphasparse_status_t rot_s_plain(const ALPHA_INT nz, float* x, const ALPHA_INT* indx, float* y, const float c, const float s);
alphasparse_status_t sctr_s_plain(const ALPHA_INT nz, const float* x, const ALPHA_INT* indx, float* y);
float doti_s_plain(const ALPHA_INT nz,  const float* x,  const ALPHA_INT* indx, const float* y);


