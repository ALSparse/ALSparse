# ALSparse

ALSparse aims to build a common interface that provides Basic Linear Algebra Subroutines for sparse computation for diverse multi-core and many-core processors, and expects to be extended on distributed and heterogeneous platforms. Recently, ALSparse is created using the basic C/C++ programming language and can be deployed on both CPU (ARMv8-based and x86-based multi-core platforms) and DCU (HIP-based many-core platform).

It is primarily constrcuted with few dependencies on third-party library and very easy to use. Users can extend the library by adding customized BLAS kernels or sparse matrix storage formats. It's possible to extend more hardware platform as well.

# Documentation

The latest ALSparse documentation and API description can be found [here](https://alphasparse.github.io/AlphaSparse/build/html/index.html).

# Supported Backends

## CPU side

ALSparse supports multiple CPU hardware platforms. Since it's written in C/C++, it can be used on nearly all kind of CPUs (The general kernels and relevant api comprise plain version). Apart from plain  kernels, high performance kernels targeting **hygon CPU** and **Kunpeng CPU** are provided as well. The fast kernels can be compiled either on x86_64 or arm based CPUs.

Up to now, the fast kernels are verified on:

- Hygon 1st gen CPU (**Zen1** based)
- Kunpeng 920 CPU(**Taishan** based)

## GPU side

ALSparse supports Deep Computing Unit platform (DCU, By Hygon) which is compatible with **HIP**. 

GPU kernels are supported on:

- DCU

# Requirements

## Mininum requirements

* GCC (9.0 or later)
* OpenMP
* Make

## Optional requirements

- MKL
- ROCM 3.9 or later
- rocSparse

For DCU users, ROCm environment is compulsive.

On CPU side , the correctness check of fast kernels are done by comparing to plain kernels. On x86_64 platform, mkl can take the place of the plain kernels.

On DCU side, rocSparse can be used to perform the corretness check.

# Quickstart

## Build

You can build ALSparse using the following steps

```
# Clone ALSparse using git
git clone https://github.com/AlphaSparse.git

# Go to ALSparse directory
cd ALSparse

# Build
PLAIN_ON=1 HIP_ON=0 HYGON_ON=0 make -j

```

Set `HIP_ON=1` if you want to build the DCU kernels, set `ARM_ON=1`  for arm kernels , and set `HYGON_ON=1`  for hygon cpu kernels.

# Tests

ALSparse provides some tests when the project is built. To run these tests, matrix has to be created or input an existing one with path.

```
# Go to ALSparse 
cd ALSparse

# Create a directory for testing matrices
mkdir Matrix

# Create test matrix and store it in ./Matrix
./bin/sparse_gen 1000

# Specify test options:
   --help       - prints help message
   --thread-num - # of threads to run kernels
   --iter       - # of test iterations
   --typeA      - matrix A type
   --data-file  - data file path of matrix
More options are listed in AlphaSparse/src/util/args.c

# Run the test, e.g.
./bin/mv_s_csr_arm_test --data-file=Matrix/1000_1000_5000.mtx [More options]
```

# License

The LICENSE file can be found in the main repository.
