#include "alphasparse/handle.h"
#include <hip/hip_runtime.h>

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

#define ROW_BITS 32
#define BLOCK_SIZE 1024
#define BLOCK_MULTIPLIER 3
#define ROWS_FOR_VECTOR 1
#define WG_BITS 24
#define WG_SIZE 256

static size_t alpha_distance(unsigned long long* a, unsigned long long* b)
{
    size_t num = 0;
    while(a++ != b) num ++;
    return num;
}

__attribute__((unused)) static unsigned int flp2(unsigned int x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x - (x >> 1);
}

static unsigned long long numThreadsForReduction(unsigned long long num_rows)
{
#if defined(__INTEL_COMPILER)
    return WG_SIZE >> (_bit_scan_reverse(num_rows - 1) + 1);
#elif(defined(__clang__) && __has_builtin(__builtin_clz)) \
    || !defined(__clang) && defined(__GNUG__)             \
           && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
    return (WG_SIZE >> (8 * sizeof(int) - __builtin_clz(num_rows - 1)));
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
    unsigned long long bit_returned;
    _BitScanReverse(&bit_returned, (num_rows - 1));
    return WG_SIZE >> (bit_returned + 1);
#else
    return flp2(WG_SIZE / num_rows);
#endif
}

static inline void ComputeRowBlocks(unsigned long long* rowBlocks,
                                    size_t&             rowBlockSize,
                                    const ALPHA_INT*      rowDelimiters,
                                    ALPHA_INT             nRows,
                                    bool                allocate_row_blocks)
{
    unsigned long long* rowBlocksBase;

    // Start at one because of rowBlock[0]
    ALPHA_INT total_row_blocks = 1;

    if(allocate_row_blocks)
    {
        rowBlocksBase = rowBlocks;
        *rowBlocks    = 0;
        ++rowBlocks;
    }

    unsigned long long sum = 0;
    unsigned long long i;
    unsigned long long last_i = 0;
    double rb2 = pow(2, ROW_BITS);
    ALPHA_INT rb2I = (ALPHA_INT)rb2;

    // Check to ensure nRows can fit in 32 bits
    if(nRows > rb2I)
    {
        fprintf(stderr, "nrow does not fit in 32 bits\n");
        exit(1);
    }

    ALPHA_INT consecutive_long_rows = 0;
    for(i = 1; i <= nRows; ++i)
    {
        ALPHA_INT row_length = (rowDelimiters[i] - rowDelimiters[i - 1]);
        sum += row_length;

        // The following section of code calculates whether you're moving between
        // a series of "short" rows and a series of "long" rows.
        // This is because the reduction in CSR-Adaptive likes things to be
        // roughly the same length. Long rows can be reduced horizontally.
        // Short rows can be reduced one-thread-per-row. Try not to mix them.
        if(row_length > 128)
        {
            ++consecutive_long_rows;
        }
        else if(consecutive_long_rows > 0)
        {
            // If it turns out we WERE in a long-row region, cut if off now.
            if(row_length < 32) // Now we're in a short-row region
            {
                consecutive_long_rows = -1;
            }
            else
            {
                consecutive_long_rows++;
            }
        }

        // If you just entered into a "long" row from a series of short rows,
        // then we need to make sure we cut off those short rows. Put them in
        // their own workgroup.
        if(consecutive_long_rows == 1)
        {
            // Assuming there *was* a previous workgroup. If not, nothing to do here.
            if(i - last_i > 1)
            {
                if(allocate_row_blocks)
                {
                    *rowBlocks = ((i - 1) << (64 - ROW_BITS));

                    // If this row fits into CSR-Stream, calculate how many rows
                    // can be used to do a parallel reduction.
                    // Fill in the low-order bits with the numThreadsForRed
                    if(((i - 1) - last_i) > ROWS_FOR_VECTOR)
                    {
                        *(rowBlocks - 1) |= numThreadsForReduction((i - 1) - last_i);
                    }

                    ++rowBlocks;
                }

                ++total_row_blocks;
                last_i = i - 1;
                sum    = row_length;
            }
        }
        else if(consecutive_long_rows == -1)
        {
            // We see the first short row after some long ones that
            // didn't previously fill up a row block.
            if(allocate_row_blocks)
            {
                *rowBlocks = ((i - 1) << (64 - ROW_BITS));
                if(((i - 1) - last_i) > ROWS_FOR_VECTOR)
                {
                    *(rowBlocks - 1) |= numThreadsForReduction((i - 1) - last_i);
                }

                ++rowBlocks;
            }

            ++total_row_blocks;
            last_i                = i - 1;
            sum                   = row_length;
            consecutive_long_rows = 0;
        }

        // Now, what's up with this row? What did it do?

        // exactly one row results in non-zero elements to be greater than blockSize
        // This is csr-vector case; bottom WGBITS == workgroup ID
        if((i - last_i == 1) && sum > BLOCK_SIZE)
        {
            ALPHA_INT numWGReq = (ALPHA_INT)ceil((double)(row_length / (BLOCK_MULTIPLIER * BLOCK_SIZE)));

            // Check to ensure #workgroups can fit in WGBITS bits, if not
            // then the last workgroup will do all the remaining work
            numWGReq = numWGReq < rb2I ? numWGReq : rb2I;

            if(allocate_row_blocks)
            {
                for(ALPHA_INT w = 1; w < numWGReq; ++w)
                {
                    *rowBlocks = ((i - 1) << (64 - ROW_BITS));
                    *rowBlocks |= (unsigned long long)(w);
                    ++rowBlocks;
                }

                *rowBlocks = (i << (64 - ROW_BITS));
                ++rowBlocks;
            }

            total_row_blocks += numWGReq;
            last_i                = i;
            sum                   = 0;
            consecutive_long_rows = 0;
        }
        // more than one row results in non-zero elements to be greater than blockSize
        // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
        else if((i - last_i > 1) && sum > BLOCK_SIZE)
        {
            // This row won't fit, so back off one.
            --i;

            if(allocate_row_blocks)
            {
                *rowBlocks = (i << (64 - ROW_BITS));
                if((i - last_i) > ROWS_FOR_VECTOR)
                {
                    *(rowBlocks - 1) |= numThreadsForReduction(i - last_i);
                }

                ++rowBlocks;
            }

            ++total_row_blocks;
            last_i                = i;
            sum                   = 0;
            consecutive_long_rows = 0;
        }
        // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
        else if(sum == BLOCK_SIZE)
        {
            if(allocate_row_blocks)
            {
                *rowBlocks = (i << (64 - ROW_BITS));
                if((i - last_i) > ROWS_FOR_VECTOR)
                {
                    *(rowBlocks - 1) |= numThreadsForReduction(i - last_i);
                }

                ++rowBlocks;
            }

            ++total_row_blocks;
            last_i                = i;
            sum                   = 0;
            consecutive_long_rows = 0;
        }
    }

    // If we didn't fill a row block with the last row, make sure we don't lose it.
    if(allocate_row_blocks && (*(rowBlocks - 1) >> (64 - ROW_BITS)) != nRows)
    {
        *rowBlocks = (unsigned long long)nRows << (64 - ROW_BITS);
        if((nRows - last_i) > ROWS_FOR_VECTOR)
        {
            *(rowBlocks - 1) |= numThreadsForReduction(i - last_i);
        }

        ++rowBlocks;
    }

    ++total_row_blocks;

    if(allocate_row_blocks)
    {
        size_t dist = alpha_distance(rowBlocksBase, rowBlocks);
        assert((2 * dist) <= rowBlockSize);
        // Update the size of rowBlocks to reflect the actual amount of memory used
        // We're multiplying the size by two because the extended precision form of
        // CSR-Adaptive requires more space for the final global reduction.
        rowBlockSize = 2 * dist;
    }
    else
    {
        rowBlockSize = 2 * total_row_blocks;
    }
}

alphasparse_status_t ONAME(alphasparse_dcu_handle_t     handle,
                        alphasparse_operation_t        trans,
                        ALPHA_INT                       m,
                        ALPHA_INT                       n,
                        ALPHA_INT                       nnz,
                        const struct alpha_dcu_matrix_descr *descr,
                        const ALPHA_Number*             csr_val,
                        const ALPHA_INT*                csr_row_ptr,
                        const ALPHA_INT*                csr_col_ind,
                        alphasparse_dcu_mat_info_t     info)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }
    else if(descr == NULL)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }
    else if(info == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Logging
    // log_trace(handle,
    //           "alphasparse_dcu_csrmv_analysis",
    //           trans,
    //           m,
    //           n,
    //           nnz,
    //           (const void*&)descr,
    //           (const void*&)csr_val,
    //           (const void*&)csr_row_ptr,
    //           (const void*&)csr_col_ind,
    //           (const void*&)info);
    // Check index base
    if(descr->base != ALPHA_SPARSE_INDEX_BASE_ZERO && descr->base != ALPHA_SPARSE_INDEX_BASE_ONE)
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    if(descr->type != ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
    {
        // TODO
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return ALPHA_SPARSE_STATUS_SUCCESS;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || csr_col_ind == nullptr || csr_val == nullptr)
    {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Clear csrmv info
    (alphasparse_dcu_destroy_csrmv_info(info->csrmv_info));

    // Create csrmv info
    (alphasparse_dcu_create_csrmv_info(&info->csrmv_info, true));

    // Stream
    hipStream_t stream = handle->stream;

    // row blocks size
    info->csrmv_info->size = 0;

    // Temporary arrays to hold device data
    ALPHA_INT * hptr = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT)*(m + 1));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        hptr, csr_row_ptr, sizeof(ALPHA_INT) * (m + 1), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Determine row blocks array size
    ComputeRowBlocks((unsigned long long*)nullptr, info->csrmv_info->size, hptr, m, false);

    // Create row blocks structure
    unsigned long long* row_blocks = (unsigned long long*)(alpha_malloc)(sizeof(unsigned long long)*info->csrmv_info->size);
    memset(row_blocks, 0, sizeof(unsigned long long)*info->csrmv_info->size);

    ComputeRowBlocks(row_blocks, info->csrmv_info->size, hptr, m, true);

    // Allocate memory on device to hold csrmv info, if required
    if(info->csrmv_info->size > 0)
    {
        THROW_IF_HIP_ERROR(hipMalloc((void**)&info->csrmv_info->row_blocks,
                                      sizeof(unsigned long long) * info->csrmv_info->size));

        // Copy row blocks information to device
        THROW_IF_HIP_ERROR(hipMemcpyAsync(info->csrmv_info->row_blocks,
                                           row_blocks,
                                           sizeof(unsigned long long) * info->csrmv_info->size,
                                           hipMemcpyHostToDevice,
                                           stream));

        // Wait for device transfer to finish
        THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }

    // Store some pointers to verify correct execution
    info->csrmv_info->trans       = trans;
    info->csrmv_info->m           = m;
    info->csrmv_info->n           = n;
    info->csrmv_info->nnz         = nnz;
    info->csrmv_info->descr       = descr;
    info->csrmv_info->csr_row_ptr = csr_row_ptr;
    info->csrmv_info->csr_col_ind = csr_col_ind;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif /*__cplusplus */