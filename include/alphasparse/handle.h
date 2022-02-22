#pragma once
#include <hip/hip_runtime_api.h>
#include <vector>

/**
 *  @brief header for the internal sparse matrix definitions
 */
#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include "spdef.h"
#include "types.h"

typedef struct ihipStream_t* hipStream_t;

typedef struct
{
    // device id
    ALPHA_INT device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    ALPHA_INT wavefront_size;
    // asic revision
    ALPHA_INT asic_rev;
    // stream ; default stream is system stream NULL
    hipStream_t stream;
    // pointer mode ; default mode is host
    alphasparse_dcu_pointer_mode_t pointer_mode;
    // logging mode
    alphasparse_dcu_layer_mode_t layer_mode;
    // device buffer
    size_t buffer_size;
    void*  buffer;
    // device one
    float*  sone;
    double* done;
    // device complex one
    ALPHA_Complex8*  cone;
    ALPHA_Complex16* zone;

    // for check
    bool check;
} alphasparse_dcu_handle;

typedef alphasparse_dcu_handle *alphasparse_dcu_handle_t;

/********************************************************************************
 * alphasparse_dcu_hyb_mat is a structure holding the alphasparse_dcu HYB matrix.
 * It must be initialized using alphasparse_dcu_create_hyb_mat() and the returned
 * handle must be passed to all subsequent library function calls that involve
 * the HYB matrix.
 * It should be destroyed at the end using alphasparse_dcu_destroy_hyb_mat().
 *******************************************************************************/
typedef struct 
{
    // num rows
    ALPHA_INT m;
    // num cols
    ALPHA_INT n;

    // partition type
    alphasparse_dcu_hyb_partition_t partition;

    // ELL matrix part
    ALPHA_INT  ell_nnz;
    ALPHA_INT  ell_width;
    ALPHA_INT* ell_col_ind;
    void*    ell_val;

    // COO matrix part
    ALPHA_INT  coo_nnz;
    ALPHA_INT* coo_row_ind;
    ALPHA_INT* coo_col_ind;
    void*    coo_val;
}_alphasparse_dcu_hyb_mat;

struct _alphasparse_dcu_trm_info
{
    // maximum non-zero entries per row
    ALPHA_INT max_nnz ;

    // device array to hold row permutation
    ALPHA_INT* row_map ;
    // device array to hold pointer to diagonal entry
    ALPHA_INT* trm_diag_ind ;
    // device pointers to hold transposed data
    ALPHA_INT* trmt_perm    ;
    ALPHA_INT* trmt_row_ptr ;
    ALPHA_INT* trmt_col_ind ;

    // some data to verify correct execution
    ALPHA_INT               m;
    ALPHA_INT               nnz;
    const struct alpha_dcu_matrix_descr * descr;
    const ALPHA_INT*        trm_row_ptr;
    const ALPHA_INT*        trm_col_ind;
};

/********************************************************************************
 * alphasparse_dcu_csrmv_info is a structure holding the alphasparse_dcu csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * alphasparse_dcu_create_csrmv_info() routine. It should be destroyed at the end
 * alphasparse_dcu_destroy_csrmv_info().
 *******************************************************************************/
enum csrgemv_algo
{
    NONE_ALGO,
    SCALAR,         // one thread process one row
    VECTOR,         // one wavefront process one row
    VECTOR_MEMALIGN,// one wavefront process one row with memory align
    ROW_PARTITION,  // Assign consecutive rows to a wavefront
    ADAPTIVE        // csr-adaptive
};

struct _alphasparse_dcu_csrmv_info
{
    // algo tune
    bool algo_tuning;
    csrgemv_algo algo;
    ALPHA_INT iter;

    // data struct for csr-adaptive method
    size_t size ;                     //num row blocks
    unsigned long long* row_blocks ;  // row blocks
    bool csr_adaptive_has_tuned;

    // some data to verify correct execution
    alphasparse_operation_t         trans;
    ALPHA_INT               m;
    ALPHA_INT               n;
    ALPHA_INT               nnz;
    const struct alpha_dcu_matrix_descr * descr;
    const void*                 csr_row_ptr;
    const void*                 csr_col_ind;
};

/********************************************************************************
 * alphasparse_dcu_csrgemm_info is a structure holding the alphasparse_dcu csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the alphasparse_dcu_create_csrgemm_info() routine. It should be destroyed at the
 * end using alphasparse_dcu_destroy_csrgemm_info().
 *******************************************************************************/
struct _alphasparse_dcu_csrgemm_info
{
    // Perform alpha * A * B
    bool mul;
    // Perform beta * D
    bool add;
};

/*! typedefs to opaque info structs */
typedef struct _alphasparse_dcu_mat_info*     alphasparse_dcu_mat_info_t;
typedef struct _alphasparse_dcu_trm_info*     alphasparse_dcu_trm_info_t;
typedef struct _alphasparse_dcu_csrmv_info*   alphasparse_dcu_csrmv_info_t;
typedef struct _alphasparse_dcu_csrgemm_info* alphasparse_dcu_csrgemm_info_t;
typedef _alphasparse_dcu_hyb_mat*      alphasparse_dcu_hyb_mat_t;

/********************************************************************************
 * alphasparse_dcu_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * alphasparse_dcu_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using alphasparse_dcu_destroy_mat_info().
 *******************************************************************************/
struct _alphasparse_dcu_mat_info
{
    // info structs
    alphasparse_dcu_trm_info_t bsrsv_upper_info  ;
    alphasparse_dcu_trm_info_t bsrsv_lower_info  ;
    alphasparse_dcu_trm_info_t bsrsvt_upper_info ;
    alphasparse_dcu_trm_info_t bsrsvt_lower_info ;
    alphasparse_dcu_trm_info_t bsric0_info       ;
    alphasparse_dcu_trm_info_t bsrilu0_info      ;

    alphasparse_dcu_csrmv_info_t   csrmv_info        ;
    alphasparse_dcu_trm_info_t     csric0_info       ;
    alphasparse_dcu_trm_info_t     csrilu0_info      ;
    alphasparse_dcu_trm_info_t     csrsv_upper_info  ;
    alphasparse_dcu_trm_info_t     csrsv_lower_info  ;
    alphasparse_dcu_trm_info_t     csrsvt_upper_info ;
    alphasparse_dcu_trm_info_t     csrsvt_lower_info ;
    alphasparse_dcu_trm_info_t     csrsm_upper_info  ;
    alphasparse_dcu_trm_info_t     csrsm_lower_info  ;
    alphasparse_dcu_trm_info_t     csrsmt_upper_info ;
    alphasparse_dcu_trm_info_t     csrsmt_lower_info ;
    alphasparse_dcu_csrgemm_info_t csrgemm_info      ;

    // zero pivot for csrsv, csrsm, csrilu0, csric0
    ALPHA_INT* zero_pivot ;

    // numeric boost for ilu0
    int         boost_enable        ;
    int         use_double_prec_tol ;
    const void* boost_tol           ;
    const void* boost_val           ;
};

/********************************************************************************
 * alphasparse_dcu_csrmv_info is a structure holding the alphasparse_dcu csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * alphasparse_dcu_create_csrmv_info() routine. It should be destroyed at the end
 * using alphasparse_dcu_destroy_csrmv_info().
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_create_csrmv_info(alphasparse_dcu_csrmv_info_t* info, bool algo_tunning);

/********************************************************************************
 * Destroy csrmv info.
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_destroy_csrmv_info(alphasparse_dcu_csrmv_info_t info);

/********************************************************************************
 * alphasparse_dcu_trm_info is a structure holding the alphasparse_dcu bsrsv, csrsv,
 * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
 * csrilu0_analysis and csric0_analysis. It must be initialized using the
 * alphasparse_dcu_create_trm_info() routine. It should be destroyed at the end
 * using alphasparse_dcu_destroy_trm_info().
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_create_trm_info(alphasparse_dcu_trm_info_t* info);

/********************************************************************************
 * Destroy trm info.
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_destroy_trm_info(alphasparse_dcu_trm_info_t info);

/********************************************************************************
 * alphasparse_dcu_check_trm_shared checks if the given trm info structure
 * shares its meta data with another trm info structure.
 *******************************************************************************/
bool alphasparse_dcu_check_trm_shared(const alphasparse_dcu_mat_info_t info, alphasparse_dcu_trm_info_t trm);

/********************************************************************************
 * alphasparse_dcu_csrgemm_info is a structure holding the alphasparse_dcu csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the alphasparse_dcu_create_csrgemm_info() routine. It should be destroyed at the
 * end using alphasparse_dcu_destroy_csrgemm_info().
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_create_csrgemm_info(alphasparse_dcu_csrgemm_info_t* info);

/********************************************************************************
 * Destroy csrgemm info.
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_destroy_csrgemm_info(alphasparse_dcu_csrgemm_info_t info);

alphasparse_status_t get_alphasparse_dcu_status_for_hip_status(hipError_t status);

/********************************************************************************
 * ELL format indexing
 *******************************************************************************/
#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

struct _alphasparse_dcu_spvec_descr
{
    bool init ;

    int64_t size;
    int64_t nnz;

    void* idx_data;
    void* val_data;

    alphasparse_dcu_indextype_t idx_type;
    alphasparse_datatype_t  data_type;

    alphasparse_index_base_t idx_base;
};

struct _alphasparse_dcu_spmat_descr
{
    bool init     ;
    bool analysed ;

    int64_t rows;
    int64_t cols;
    int64_t nnz;

    void* row_data;
    void* col_data;
    void* ind_data;
    void* val_data;

    alphasparse_dcu_indextype_t row_type;
    alphasparse_dcu_indextype_t col_type;
    alphasparse_datatype_t      data_type;

    alphasparse_index_base_t idx_base;
    alphasparse_format_t     format;

    struct alpha_dcu_matrix_descr  * descr;
    alphasparse_dcu_mat_info_t  info;
};

struct _alphasparse_dcu_dnvec_descr
{
    bool init ;

    int64_t            size;
    void*              values;
    alphasparse_datatype_t data_type;
};

struct _alphasparse_dcu_dnmat_descr
{
    bool init ;

    int64_t rows;
    int64_t cols;
    int64_t ld;

    void* values;

    alphasparse_datatype_t data_type;
    alphasparse_layout_t   order;
};

typedef struct _alphasparse_dcu_spvec_descr* alphasparse_dcu_spvec_descr_t;
typedef struct _alphasparse_dcu_spmat_descr* alphasparse_dcu_spmat_descr_t;
typedef struct _alphasparse_dcu_dnvec_descr* alphasparse_dcu_dnvec_descr_t;
typedef struct _alphasparse_dcu_dnmat_descr* alphasparse_dcu_dnmat_descr_t;

alphasparse_status_t alphasparse_dcu_create_mat_descr(alpha_dcu_matrix_descr_t * descr);
alphasparse_status_t alphasparse_dcu_create_mat_info(alphasparse_dcu_mat_info_t * info);
alphasparse_status_t alphasparse_dcu_destroy_mat_descr(alpha_dcu_matrix_descr_t descr);
alphasparse_status_t alphasparse_dcu_destroy_mat_info(alphasparse_dcu_mat_info_t info);
alphasparse_status_t alphasparse_dcu_destroy_trm_info(alphasparse_dcu_trm_info_t info);
alphasparse_status_t alphasparse_dcu_get_handle(alphasparse_dcu_handle_t *handle);
alphasparse_status_t alphasparse_dcu_destory_handle(alphasparse_dcu_handle_t handle);
alphasparse_status_t init_handle(alphasparse_dcu_handle_t *handle);

double get_time_us(void);

double get_avg_time(std::vector<double> times);

void alphasparse_dcu_init_s_csr_laplace2d(ALPHA_INT*      row_ptr,
                                  ALPHA_INT*             col_ind,
                                  float*               val,
                                  ALPHA_INT              dim_x,
                                  ALPHA_INT              dim_y,
                                  ALPHA_INT&             M,
                                  ALPHA_INT&             N,
                                  ALPHA_INT&             nnz,
                                  alphasparse_index_base_t base);

void alphasparse_dcu_init_d_csr_laplace2d(ALPHA_INT*      row_ptr,
                                  ALPHA_INT*             col_ind,
                                  double*              val,
                                  ALPHA_INT              dim_x,
                                  ALPHA_INT              dim_y,
                                  ALPHA_INT&             M,
                                  ALPHA_INT&             N,
                                  ALPHA_INT&             nnz,
                                  alphasparse_index_base_t base);

void alphasparse_dcu_init_c_csr_laplace2d(ALPHA_INT*      row_ptr,
                                  ALPHA_INT*             col_ind,
                                  ALPHA_Complex8*        val,
                                  ALPHA_INT              dim_x,
                                  ALPHA_INT              dim_y,
                                  ALPHA_INT&             M,
                                  ALPHA_INT&             N,
                                  ALPHA_INT&             nnz,
                                  alphasparse_index_base_t base);

void alphasparse_dcu_init_z_csr_laplace2d(ALPHA_INT*      row_ptr,
                                  ALPHA_INT*             col_ind,
                                  ALPHA_Complex16*        val,
                                  ALPHA_INT              dim_x,
                                  ALPHA_INT              dim_y,
                                  ALPHA_INT&             M,
                                  ALPHA_INT&             N,
                                  ALPHA_INT&             nnz,
                                  alphasparse_index_base_t base);

#ifdef __cplusplus
}
#endif /*__cplusplus */
