#pragma once

alphasparse_status_t axpy_d_plain(const ALPHA_INT nz,  const double a,  const double* x,  const ALPHA_INT* indx,  double* y);
alphasparse_status_t gthr_d_plain(const ALPHA_INT nz,	const double* y, double* x, const ALPHA_INT* indx);
alphasparse_status_t gthrz_d_plain(const ALPHA_INT nz, double* y, double* x, const ALPHA_INT* indx);
alphasparse_status_t rot_d_plain(const ALPHA_INT nz, double* x, const ALPHA_INT* indx, double* y, const double c, const double s);
alphasparse_status_t sctr_d_plain(const ALPHA_INT nz, const double* x, const ALPHA_INT* indx, double* y);
double doti_d_plain(const ALPHA_INT nz,  const double* x,  const ALPHA_INT* indx, const double* y);