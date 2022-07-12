#pragma once

/**
 * @brief header for basic types and constants for openspblas spblas API
 */

/* status of the routines */
typedef enum
{
    ALPHA_SPARSE_STATUS_SUCCESS = 0,          /* the operation was successful */
    ALPHA_SPARSE_STATUS_NOT_INITIALIZED = 1,  /* empty handle or matrix arrays */
    ALPHA_SPARSE_STATUS_ALLOC_FAILED = 2,     /* internal error: memory allocation failed */
    ALPHA_SPARSE_STATUS_INVALID_VALUE = 3,    /* invalid input value */
    ALPHA_SPARSE_STATUS_EXECUTION_FAILED = 4, /* e.g. 0-diagonal element for triangular solver, etc. */
    ALPHA_SPARSE_STATUS_INTERNAL_ERROR = 5,   /* internal error */
    ALPHA_SPARSE_STATUS_NOT_SUPPORTED = 6,    /* e.g. operation for double precision doesn't support other types */
    ALPHA_SPARSE_STATUS_INVALID_POINTER = 7,  /* e.g. invlaid pointers for DCU */
    ALPHA_SPARSE_STATUS_INVALID_HANDLE = 8    /* e.g. invlaid pointers for DCU */
} alphasparse_status_t;

/* sparse matrix operations */
typedef enum
{
    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE = 0,
    ALPHA_SPARSE_OPERATION_TRANSPOSE = 1,
    ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} alphasparse_operation_t;

#define ALPHA_SPARSE_OPERATION_NUM 3

/* supported matrix types */
typedef enum
{
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL = 0,   /*    General case                    */
    ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC = 1, /*    Triangular part of              */
    ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN = 2, /*    the matrix is to be processed   */
    ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR = 3,
    ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL = 4, /* diagonal matrix; only diagonal elements will be processed */
    ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 5,
    ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 6 /* block-diagonal matrix; only diagonal blocks will be processed */
} alphasparse_matrix_type_t;

#define ALPHA_SPARSE_MATRIX_TYPE_NUM 2

/* sparse matrix indexing: C-style or Fortran-style */
typedef enum
{
    ALPHA_SPARSE_INDEX_BASE_ZERO = 0, /* C-style */
    ALPHA_SPARSE_INDEX_BASE_ONE = 1   /* Fortran-style */
} alphasparse_index_base_t;

#define ALPHA_SPARSE_INDEX_NUM 2

/* applies to triangular matrices only ( ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC, ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN, ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR ) */
typedef enum
{
    ALPHA_SPARSE_FILL_MODE_LOWER = 0, /* lower triangular part of the matrix is stored */
    ALPHA_SPARSE_FILL_MODE_UPPER = 1, /* upper triangular part of the matrix is stored */
} alphasparse_fill_mode_t;

#define ALPHA_SPARSE_FILL_MODE_NUM 2

/* applies to triangular matrices only ( ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC, ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN, ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR ) */
typedef enum
{
    ALPHA_SPARSE_DIAG_NON_UNIT = 0, /* triangular matrix with non-unit diagonal */
    ALPHA_SPARSE_DIAG_UNIT = 1      /* triangular matrix with unit diagonal */
} alphasparse_diag_type_t;

#define ALPHA_SPARSE_DIAG_TYPE_NUM 2

/* applicable for Level 3 operations with dense matrices; describes storage scheme for dense matrix (row major or column major) */
typedef enum
{
    ALPHA_SPARSE_LAYOUT_ROW_MAJOR = 0,   /* C-style */
    ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR = 1 /* Fortran-style */
} alphasparse_layout_t;

#define ALPHA_SPARSE_LAYOUT_NUM 2

/* verbose mode; if verbose mode activated, handle should collect and report profiling / optimization info */
typedef enum
{
    ALPHA_SPARSE_VERBOSE_OFF = 0,
    ALPHA_SPARSE_VERBOSE_BASIC = 1,   /* output contains high-level information about optimization algorithms, issues, etc. */
    ALPHA_SPARSE_VERBOSE_EXTENDED = 2 /* provide detailed output information */
} alpha_verbose_mode_t;

/* memory optimization hints from user: describe how much memory could be used on optimization stage */
typedef enum
{
    ALPHA_SPARSE_MEMORY_NONE = 0,      /* no memory should be allocated for matrix values and structures; auxiliary structures could be created only for workload balancing, parallelization, etc. */
    ALPHA_SPARSE_MEMORY_AGGRESSIVE = 1 /* matrix could be converted to any internal format */
} alphasparse_memory_usage_t;

typedef enum
{
    ALPHA_SPARSE_STAGE_FULL_MULT = 0,
    ALPHA_SPARSE_STAGE_NNZ_COUNT = 1,
    ALPHA_SPARSE_STAGE_FINALIZE_MULT = 2,
    ALPHA_SPARSE_STAGE_FULL_MULT_NO_VAL = 3,
    ALPHA_SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 4
} alphasparse_request_t;

/*************************************************************************************************/
/*** Opaque structure for sparse matrix in internal format, further D - means double precision ***/
/*************************************************************************************************/

/*
 * openspblas sparse_matrix implement
 */
// struct  sparse_matrix;
// typedef struct sparse_matrix *alphasparse_matrix_t;

/*
 * openspblas sparse matirx implement;
 * ----------------------------------------------------------------------------------------------------------------------
*/

typedef enum
{
    ALPHA_SPARSE_FORMAT_COO = 0,
    ALPHA_SPARSE_FORMAT_CSR = 1,
    ALPHA_SPARSE_FORMAT_CSC = 2,
    ALPHA_SPARSE_FORMAT_BSR = 3,
    ALPHA_SPARSE_FORMAT_SKY = 4,
    ALPHA_SPARSE_FORMAT_DIA = 5,
    ALPHA_SPARSE_FORMAT_ELL = 6,
    ALPHA_SPARSE_FORMAT_GEBSR = 7,
    ALPHA_SPARSE_FORMAT_HYB = 8,
    ALPHA_SPARSE_FORMAT_COO_AOS = 9,
    ALPHA_SPARSE_FORMAT_CSR5 = 10
} alphasparse_format_t;

#define ALPHA_SPARSE_FORMAT_NUM 6

typedef enum
{
    ALPHA_SPARSE_DATATYPE_FLOAT = 0,
    ALPHA_SPARSE_DATATYPE_DOUBLE = 1,
    ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX = 2,
    ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX = 3
} alphasparse_datatype_t;

#define ALPHA_SPARSE_DATATYPE_NUM 4

#ifndef COMPLEX
#ifndef DOUBLE
#define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_FLOAT
#else
#define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_DOUBLE
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX
#else
#define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX
#endif

#endif

typedef enum {
  ALPHA_SPARSE_EXE_HOST = 0,
  ALPHA_SPARSE_EXE_DEVICE = 1,
} alphasparse_executor_t;

typedef void *alpha_internal_spmat;

typedef struct {
  alpha_internal_spmat mat;
  alphasparse_format_t format;        // csr,coo,csc,bsr,ell,dia,sky...
  alphasparse_datatype_t datatype;    // s,d,c,z
  void *inspector;  // for autotuning
  void *dcu_info;                     // for dcu autotuning, alphasparse_dcu_mat_info_t
  alphasparse_executor_t exe;
} alphasparse_matrix;

typedef alphasparse_matrix *alphasparse_matrix_t;
/*
 * ----------------------------------------------------------------------------------------------------------------------
 */

/* descriptor of main sparse matrix properties */
struct alpha_matrix_descr
{
    alphasparse_matrix_type_t type; /* matrix type: general, diagonal or triangular / symmetric / hermitian */
    alphasparse_fill_mode_t mode;   /* upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case) */
    alphasparse_diag_type_t diag;   /* unit or non-unit diagonal ( for triangular / symmetric / hermitian case) */
};


/*structures for DCU usages*/
struct alpha_dcu_matrix_descr {
  alphasparse_matrix_type_t
      type; /* matrix type: general, diagonal or triangular / symmetric / hermitian */
  alphasparse_fill_mode_t mode; /* upper or lower triangular part of the matrix ( for triangular /
                                  symmetric / hermitian case) */
  alphasparse_diag_type_t
      diag; /* unit or non-unit diagonal ( for triangular / symmetric / hermitian case) */
  alphasparse_index_base_t base; /* C-style or Fortran-style*/
};

typedef struct alpha_dcu_matrix_descr *alpha_dcu_matrix_descr_t;
typedef enum {
  ALPHA_SPARSE_DCU_POINTER_MODE_HOST = 0,
  ALPHA_SPARSE_DCU_POINTER_MODE_DVICE = 1,
} alphasparse_dcu_pointer_mode_t;

typedef enum {
  ALPHA_SPARSE_DCU_LAYER_MODE_NONE = 0,      /**< layer is not active. */
  ALPHA_SPARSE_DCU_LAYER_MODE_LOG_TRACE = 1, /**< layer is in logging mode. */
  ALPHA_SPARSE_DCU_LAYER_MODE_LOG_BENCH = 2  /**< layer is in benchmarking mode. */
} alphasparse_dcu_layer_mode_t;

typedef enum {
  ALPHA_SPARSE_DCU_INDEXTYPE_U16 = 1, /**< 16 bit unsigned integer. */
  ALPHA_SPARSE_DCU_INDEXTYPE_I32 = 2, /**< 32 bit signed integer. */
  ALPHA_SPARSE_DCU_INDEXTYPE_I64 = 3  /**< 64 bit signed integer. */
} alphasparse_dcu_indextype_t;

typedef enum {
  ALPHA_SPARSE_DCU_HYB_PARTITION_AUTO = 0, /**< automatically decide on ELL nnz per row. */
  ALPHA_SPARSE_DCU_HYB_PARTITION_USER = 1, /**< user given ELL nnz per row. */
  ALPHA_SPARSE_DCU_HYB_PARTITION_MAX = 2   /**< max ELL nnz per row, no COO part. */
} alphasparse_dcu_hyb_partition_t;
typedef enum {
  ALPHA_SPARSE_DCU_ANALYSIS_POLICY_REUSE = 0, /**< try to re-use meta data. */
  ALPHA_SPARSE_DCU_ANALYSIS_POLICY_FORCE = 1  /**< force to re-build meta data. */
} alphasparse_dcu_analysis_policy_t;

typedef enum {
  ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO = 0 /**< automatically decide on level information. */
} alphasparse_dcu_solve_policy_t;

typedef enum {
  ALPHA_SPARSE_DCU_ACTION_SYMBOLIC = 0, /**< Operate only on indices. */
  ALPHA_SPARSE_DCU_ACTION_NUMERIC = 1   /**< Operate on data and indices. */
} alphasparse_dcu_action_t;

typedef enum {
  ALPHA_SPARSE_DCU_DENSE_TO_SPARSE_ALG_DEFAULT =
      0, /**< Default dense to sparse algorithm for the given format. */
} alphasparse_dcu_dense_to_sparse_alg_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPARSE_TO_DENSE_ALG_DEFAULT =
      0, /**< Default sparse to dense algorithm for the given format. */
} alphasparse_dcu_sparse_to_dense_alg_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPMV_ALG_DEFAULT = 0, /**< Default SpMV algorithm for the given format. */
  ALPHA_SPARSE_DCU_SPMV_ALG_COO = 1,     /**< COO SpMV algorithm for COO matrices. */
  ALPHA_SPARSE_DCU_SPMV_ALG_CSR_ADAPTIVE =
      2,                                  /**< CSR SpMV algorithm 1 (adaptive) for CSR matrices. */
  ALPHA_SPARSE_DCU_SPMV_ALG_CSR_STREAM = 3, /**< CSR SpMV algorithm 2 (stream) for CSR matrices. */
  ALPHA_SPARSE_DCU_SPMV_ALG_ELL = 4         /**< ELL SpMV algorithm for ELL matrices. */
} alphasparse_dcu_spmv_alg_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_AUTO = 0,        /**< Automatic stage detection. */
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_BUFFER_SIZE = 1, /**< Returns the required buffer size. */
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_NNZ = 2,         /**< Computes number of non-zero entries. */
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_COMPUTE = 3      /**< Performs the actual SpGEMM computation. */
} alphasparse_dcu_spgemm_stage_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPGEMM_ALG_DEFAULT = 0 /**< Default SpGEMM algorithm for the given format. */
} alphasparse_dcu_spgemm_alg_t;
