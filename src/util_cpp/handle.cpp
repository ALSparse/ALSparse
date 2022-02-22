#include "alphasparse/handle.h"
#include <hip/hip_runtime.h>
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/error.h"
#include "alphasparse/util/malloc.h"
#include <math.h>
#include <assert.h>
#include <memory.h>
#include <sys/time.h>

__global__ void init_kernel(){};

alphasparse_status_t get_alphasparse_dcu_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
    // success
    case hipSuccess:
        return ALPHA_SPARSE_STATUS_SUCCESS;

    // internal hip memory allocation
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return ALPHA_SPARSE_STATUS_ALLOC_FAILED;

    // user-allocated hip memory
    case hipErrorInvalidDevicePointer: // hip memory
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;

    // user-allocated device, stream, event
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;

    // library using hip incorrectly
    case hipErrorInvalidValue:
        return ALPHA_SPARSE_STATUS_INTERNAL_ERROR;

    // hip runtime failing
    case hipErrorNoDevice: // no hip devices
    case hipErrorUnknown:
    default:
        return ALPHA_SPARSE_STATUS_INTERNAL_ERROR;
    }
}

alphasparse_status_t init_handle(alphasparse_dcu_handle_t *handle)
{
    *handle = (alphasparse_dcu_handle_t)alpha_malloc(sizeof(alphasparse_dcu_handle));
    (*handle)->stream = 0;
    (*handle)->pointer_mode = ALPHA_SPARSE_DCU_POINTER_MODE_HOST;

    (*handle)->check = false;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t alphasparse_dcu_get_handle(alphasparse_dcu_handle_t *handle)
{
    // Default device is active device
    if((*handle) == nullptr) init_handle(handle); 
    THROW_IF_HIP_ERROR(hipGetDevice(&(*handle)->device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&(*handle)->properties, (*handle)->device));

    // Device wavefront size
    (*handle)->wavefront_size = (*handle)->properties.warpSize;

#if HIP_VERSION >= 307
    // ASIC revision
    (*handle)->asic_rev = (*handle)->properties.asicRevision;
#else
    (*handle)->asic_rev = 0;
#endif

    // Layer mode
    char* str_layer_mode;
    if((str_layer_mode = getenv("ALPHA_SPARSE_DCU_LAYER")) == NULL)
    {
        (*handle)->layer_mode = ALPHA_SPARSE_DCU_LAYER_MODE_NONE;
    }
    else
    {
        (*handle)->layer_mode = (alphasparse_dcu_layer_mode_t)(atoi(str_layer_mode));
    }

    // Obtain size for coomv device buffer
    ALPHA_INT nthreads = (*handle)->properties.maxThreadsPerBlock;
    ALPHA_INT nprocs   = (*handle)->properties.multiProcessorCount;
    ALPHA_INT nblocks  = (nprocs * nthreads - 1) / 128 + 1;
    ALPHA_INT nwfs     = nblocks * (128 / (*handle)->properties.warpSize);

    size_t coomv_size = (((sizeof(ALPHA_INT) + 16) * nwfs - 1) / 256 + 1) * 256;

    // Allocate device buffer
    (*handle)->buffer_size = (coomv_size > 1024 * 1024) ? coomv_size : 1024 * 1024;

    THROW_IF_HIP_ERROR(hipMalloc((void**)&(*handle)->buffer, (*handle)->buffer_size));
    // Device one
    THROW_IF_HIP_ERROR(hipMalloc((void**)&((*handle)->sone), sizeof(float)));
    THROW_IF_HIP_ERROR(hipMalloc((void**)&((*handle)->done), sizeof(double)));
    THROW_IF_HIP_ERROR(hipMalloc((void**)&((*handle)->cone), sizeof(ALPHA_Complex8)));
    THROW_IF_HIP_ERROR(hipMalloc((void**)&((*handle)->zone), sizeof(ALPHA_Complex16)));

    // Execute empty kernel for initialization
    hipLaunchKernelGGL(init_kernel, dim3(1), dim3(1), 0, (*handle)->stream);

    // Execute memset for initialization
    THROW_IF_HIP_ERROR(hipMemsetAsync((*handle)->sone, 0, sizeof(float), (*handle)->stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync((*handle)->done, 0, sizeof(double), (*handle)->stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync((*handle)->cone, 0, sizeof(ALPHA_Complex8), (*handle)->stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync((*handle)->zone, 0, sizeof(ALPHA_Complex16), (*handle)->stream));

    float  hsone = 1.0f;
    double hdone = 1.0;

    ALPHA_Complex8  hcone = {1.0f, 0.0f};
    ALPHA_Complex16 hzone = {1.0, 0.0};

    THROW_IF_HIP_ERROR(hipMemcpyAsync((*handle)->sone, &hsone, sizeof(float), hipMemcpyHostToDevice, (*handle)->stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync((*handle)->done, &hdone, sizeof(double), hipMemcpyHostToDevice, (*handle)->stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        (*handle)->cone, &hcone, sizeof(ALPHA_Complex8), hipMemcpyHostToDevice, (*handle)->stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        (*handle)->zone, &hzone, sizeof(ALPHA_Complex16), hipMemcpyHostToDevice, (*handle)->stream));

    // Wait for device transfer to finish
    THROW_IF_HIP_ERROR(hipStreamSynchronize((*handle)->stream));

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
alphasparse_status_t alphasparse_dcu_destory_handle(alphasparse_dcu_handle_t handle)
{
    PRINT_IF_HIP_ERROR(hipFree(handle->buffer));
    PRINT_IF_HIP_ERROR(hipFree(handle->sone));
    PRINT_IF_HIP_ERROR(hipFree(handle->done));
    PRINT_IF_HIP_ERROR(hipFree(handle->cone));
    PRINT_IF_HIP_ERROR(hipFree(handle->zone));

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t alphasparse_dcu_create_csrmv_info(alphasparse_dcu_csrmv_info_t* info, bool algo_tunning)
{
    if(info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    else
    {
        // Allocate
        *info = (alphasparse_dcu_csrmv_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_csrmv_info));
        
        (*info)->algo_tuning = algo_tunning;
        (*info)->algo        = NONE_ALGO;

        (*info)->size = 0;
        (*info)->csr_adaptive_has_tuned = false;
        (*info)->row_blocks = nullptr;
        
        (*info)->csr_row_ptr = nullptr;
        (*info)->csr_col_ind = nullptr;

        return ALPHA_SPARSE_STATUS_SUCCESS;
    }
}

/********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_destroy_csrmv_info(alphasparse_dcu_csrmv_info_t info)
{
    if(info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }
    // Clean up row blocks
    if(info->size > 0)
    {
        THROW_IF_HIP_ERROR(hipFree(info->row_blocks));
    }
    // Destruct
    alpha_free(info);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

/********************************************************************************
 * \brief Destroy trm info.
 *******************************************************************************/
alphasparse_status_t alphasparse_dcu_destroy_trm_info(alphasparse_dcu_trm_info_t info)
{
    if(info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Clean up
    if(info->row_map != nullptr)
    {
        THROW_IF_HIP_ERROR(hipFree(info->row_map));
        info->row_map = nullptr;
    }

    if(info->trm_diag_ind != nullptr)
    {
        THROW_IF_HIP_ERROR(hipFree(info->trm_diag_ind));
        info->trm_diag_ind = nullptr;
    }

    // Clear trmt arrays
    if(info->trmt_perm != nullptr)
    {
        THROW_IF_HIP_ERROR(hipFree(info->trmt_perm));
        info->trmt_perm = nullptr;
    }

    if(info->trmt_row_ptr != nullptr)
    {
        THROW_IF_HIP_ERROR(hipFree(info->trmt_row_ptr));
        info->trmt_row_ptr = nullptr;
    }

    if(info->trmt_col_ind != nullptr)
    {
        THROW_IF_HIP_ERROR(hipFree(info->trmt_col_ind));
        info->trmt_col_ind = nullptr;
    }

    // Destruct
    alpha_free(info);


    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t alphasparse_dcu_create_mat_descr(alpha_dcu_matrix_descr_t * descr)
{
    if(descr == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    else
    {
        *descr = nullptr;
        // Allocate
        *descr = (alpha_dcu_matrix_descr_t)alpha_malloc(sizeof(struct alpha_dcu_matrix_descr));
        (*descr)->base = ALPHA_SPARSE_INDEX_BASE_ZERO;
        (*descr)->type = ALPHA_SPARSE_MATRIX_TYPE_GENERAL;
        (*descr)->mode = ALPHA_SPARSE_FILL_MODE_LOWER;
        (*descr)->diag = ALPHA_SPARSE_DIAG_NON_UNIT;
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }
}

alphasparse_status_t alphasparse_dcu_destroy_mat_descr(alpha_dcu_matrix_descr_t descr)
{
    // Destruct
    alpha_free(descr);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t alphasparse_dcu_create_mat_info(alphasparse_dcu_mat_info_t * info)
{
    if(info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    else
    {
        *info = nullptr;
        // Allocate
        *info = (alphasparse_dcu_mat_info_t)alpha_malloc(sizeof(struct _alphasparse_dcu_mat_info));
        (*info)->bsrsv_upper_info = nullptr;
        (*info)->bsrsv_lower_info = nullptr;
        (*info)->bsrsvt_upper_info = nullptr;
        (*info)->bsrsvt_lower_info = nullptr;

        (*info)->bsric0_info = nullptr;
        (*info)->bsrilu0_info = nullptr;

        (*info)->csrmv_info = nullptr;

        (*info)->csric0_info = nullptr;
        (*info)->csrilu0_info = nullptr;

        (*info)->csrsv_upper_info = nullptr;
        (*info)->csrsv_lower_info = nullptr;
        (*info)->csrsvt_upper_info = nullptr;
        (*info)->csrsvt_lower_info = nullptr;

        (*info)->csrsm_upper_info = nullptr;
        (*info)->csrsm_lower_info = nullptr;
        (*info)->csrsmt_upper_info = nullptr;
        (*info)->csrsmt_lower_info = nullptr;

        (*info)->csrgemm_info = nullptr;

        (*info)->zero_pivot = nullptr;
        (*info)->boost_tol = nullptr;
        (*info)->boost_val = nullptr;

        (*info)->boost_enable = 0;
        (*info)->use_double_prec_tol = 0;

        return ALPHA_SPARSE_STATUS_SUCCESS;
    }
}

alphasparse_status_t alphasparse_dcu_destroy_csrgemm_info(alphasparse_dcu_csrgemm_info_t info)
{
    if(info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Destruct
    alpha_free(info);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparse_status_t alphasparse_dcu_destroy_mat_info(alphasparse_dcu_mat_info_t info)
{
    if(info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Uncouple shared meta data
    if(info->csrsv_lower_info == info->csrilu0_info || info->csrsv_lower_info == info->csric0_info
       || info->csrsv_lower_info == info->csrsm_lower_info)
    {
        info->csrsv_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsm_lower_info == info->csrilu0_info || info->csrsm_lower_info == info->csric0_info)
    {
        info->csrsm_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrilu0_info == info->csric0_info)
    {
        info->csrilu0_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsv_upper_info == info->csrsm_upper_info)
    {
        info->csrsv_upper_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsvt_lower_info == info->csrsmt_lower_info)
    {
        info->csrsvt_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsvt_upper_info == info->csrsmt_upper_info)
    {
        info->csrsvt_upper_info = nullptr;
    }

    // Clear csrmv info struct
    if(info->csrmv_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_csrmv_info(info->csrmv_info));
    }

    // Clear bsrsvt upper info struct
    if(info->bsrsvt_upper_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->bsrsvt_upper_info));
    }

    // Clear bsrsvt lower info struct
    if(info->bsrsvt_lower_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->bsrsvt_lower_info));
    }

    // Clear csrsvt upper info struct
    if(info->csrsvt_upper_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsvt_upper_info));
    }

    // Clear csrsvt lower info struct
    if(info->csrsvt_lower_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsvt_lower_info));
    }

    // Clear csrsmt upper info struct
    if(info->csrsmt_upper_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsmt_upper_info));
    }

    // Clear csrsmt lower info struct
    if(info->csrsmt_lower_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsmt_lower_info));
    }

    // Clear csric0 info struct
    if(info->csric0_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csric0_info));
    }

    // Clear csrilu0 info struct
    if(info->csrilu0_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrilu0_info));
    }

    // Clear bsrsv upper info struct
    if(info->bsrsv_upper_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->bsrsv_upper_info));
    }

    // Clear bsrsv lower info struct
    if(info->bsrsv_lower_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->bsrsv_lower_info));
    }

    // Clear csrsv upper info struct
    if(info->csrsv_upper_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsv_upper_info));
    }

    // Clear csrsv lower info struct
    if(info->csrsv_lower_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsv_lower_info));
    }

    // Clear csrsm upper info struct
    if(info->csrsm_upper_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsm_upper_info));
    }

    // Clear csrsm lower info struct
    if(info->csrsm_lower_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_trm_info(info->csrsm_lower_info));
    }

    // Clear csrgemm info struct
    if(info->csrgemm_info != nullptr)
    {
        RETURN_IF_ALPHA_SPARSE_DCU_ERROR(alphasparse_dcu_destroy_csrgemm_info(info->csrgemm_info));
    }

    // Clear zero pivot
    if(info->zero_pivot != nullptr)
    {
        THROW_IF_HIP_ERROR(hipFree(info->zero_pivot));
        info->zero_pivot = nullptr;
    }

    // Destruct
    alpha_free(info);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

double get_time_us(void)
{
    hipDeviceSynchronize();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

double get_avg_time(std::vector<double> times)
{
    double avg = 0.;
    for (int i = 0; i < times.size(); i++) {
        avg += times[i];
    }
    avg /= times.size();
    double avg2 = 0.;
    int cnt = 0;
    for (int i = 0; i < times.size(); i++) {
        if (times[i] < 2 * avg) {
            avg2 += times[i];
            cnt++;
        }
    }
    return avg2 / cnt;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */