#pragma once

#if defined(__CUDACC__)
#define PHYSICSMADE_HD __host__ __device__
#else
#define PHYSICSMADE_HD
#endif
