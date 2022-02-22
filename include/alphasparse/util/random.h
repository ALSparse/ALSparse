#pragma once

#include <stdlib.h>
#include <time.h>

inline unsigned int time_seed() {
  time_t t;
  return (unsigned)time(&t);
}

inline int random_int(int m) { return rand() % m; }
inline long long random_long(long long m) { return (long long)rand() % m; }

inline double random_double() { return (double)rand() / RAND_MAX; }

inline float random_float() { return (float)rand() / RAND_MAX; }