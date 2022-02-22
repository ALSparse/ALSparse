#include <alphasparse/opt.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "alphasparse/format.h"
#include "alphasparse/util.h"

alphasparse_status_t ONAME(const ALPHA_SPMAT_CSR *A, ALPHA_SPMAT_CSR5 **dst) {
    // if (!A->ordered) {
    //     printf("we need sorted csr.\n");
    //     return ALpPHA_SPARSE_STATUS_INVALID_VALUE;
    // }

    ALPHA_SPMAT_CSR5 *B = alpha_malloc(sizeof(ALPHA_SPMAT_CSR5));
    *dst = B;

    //init host point
    B->col_idx    = NULL;
    B->row_ptr    = NULL;
    B->val        = NULL;

    B->tile_ptr             = NULL;
    B->tile_desc            = NULL;
    B->tile_desc_offset_ptr = NULL;
    B->tile_desc_offset     = NULL;
    B->calibrator           = NULL;

    B->num_rows = A->rows; 
    B->num_cols = A->cols;
    B->nnz = A->rows_end[A->rows - 1];

    B->val     = alpha_memalign((uint64_t)(B->nnz) * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    B->row_ptr = alpha_memalign((uint64_t)(A->rows + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    B->col_idx = alpha_memalign((uint64_t)(B->nnz) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);

    for( ALPHA_INT i=0; i < B->num_rows+1; i++) {
        B->row_ptr[i] = A->rows_start[i];
    }

    // compute sigma
    int r = 4;
    int s = 32;
    int t = 256;
    int u = 6;

    int csr_nnz_per_row = B->nnz / B->num_rows;
    if (csr_nnz_per_row <= r)
        B->csr5_sigma = r;
    else if (csr_nnz_per_row > r && csr_nnz_per_row <= s)
        B->csr5_sigma = csr_nnz_per_row;
    else if (csr_nnz_per_row <= t && csr_nnz_per_row > s)
        B->csr5_sigma = s;
    else // csr_nnz_per_row > t
        B->csr5_sigma = u;

    // conversion
    // compute #bits required for `y_offset' and `scansum_offset'
    int base = 2;
    B->csr5_bit_y_offset = 1;
    while (base < ALPHA_CSR5_OMEGA * B->csr5_sigma)
    { base *= 2; B->csr5_bit_y_offset++; }

    base = 2;
    B->csr5_bit_scansum_offset = 1;
    while (base < ALPHA_CSR5_OMEGA)
    { base *= 2; B->csr5_bit_scansum_offset++; }

    if ( (size_t) B->csr5_bit_y_offset + B->csr5_bit_scansum_offset >
        sizeof(uint32_t) * 8 - 1)
    {
        printf("error: csr5-omega not supported.\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    int bit_all = B->csr5_bit_y_offset + B->csr5_bit_scansum_offset
                + B->csr5_sigma;
    B->csr5_num_packets = ceil((float)bit_all
                                /(float)(sizeof(uint32_t)*8));

    // calculate the number of tiles
    B->csr5_p = ceil((float)B->nnz
                    / (float)(ALPHA_CSR5_OMEGA * B->csr5_sigma));
    //printf("sigma = %i, p = %i\n", B->csr5_sigma, B->csr5_p);
    // malloc the newly added arrays for CSR5
    B->tile_ptr = alpha_memalign((uint64_t)(B->csr5_p+1) * sizeof(uint32_t), DEFAULT_ALIGNMENT);
    for( ALPHA_INT i=0; i<B->csr5_p+1; i++) {
        B->tile_ptr[i] = 0;
    }

    B->tile_desc = alpha_memalign((uint64_t)(B->csr5_p * ALPHA_CSR5_OMEGA * B->csr5_num_packets) * sizeof(uint32_t), DEFAULT_ALIGNMENT);
    for( ALPHA_INT i=0; i<B->csr5_p * ALPHA_CSR5_OMEGA
                            * B->csr5_num_packets; i++) {
        B->tile_desc[i] = 0;
    }

    B->calibrator = alpha_memalign((uint64_t)(B->csr5_p) * sizeof(ALPHA_Number), DEFAULT_ALIGNMENT);
    for( ALPHA_INT i=0; i<B->csr5_p; i++) {
        alpha_setzero(B->calibrator[i]);
    }

    B->tile_desc_offset_ptr = alpha_memalign((uint64_t)(B->csr5_p+1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    for( ALPHA_INT i=0; i<B->csr5_p+1; i++) {
        B->tile_desc_offset_ptr[i] = 0;
    }


    // convert csr data to csr5 data (3 steps)
    // step 1 generate tile pointer
    // step 1.1 binary search row pointer
    for (ALPHA_INT global_id = 0; global_id <= B->csr5_p;
        global_id++)
    {
        // compute tile boundaries by tile of size sigma * omega
        ALPHA_INT boundary = global_id * B->csr5_sigma
                                * ALPHA_CSR5_OMEGA;

        // clamp tile boundaries to [0, nnz]
        boundary = boundary > B->nnz ? B->nnz : boundary;

        // binary search
        ALPHA_INT start = 0, stop = B->num_rows, median;
        ALPHA_INT key_median;
        while (stop >= start)
        {
            median = (stop + start) / 2;
            key_median = B->row_ptr[median];
            if (boundary >= key_median)
                start = median + 1;
            else
                stop = median - 1;
        }
        B->tile_ptr[global_id] = start-1;
    }

    // step 1.2 check empty rows
    for (ALPHA_INT group_id = 0; group_id < B->csr5_p; group_id++) {
        int dirty = 0;

        uint32_t start = B->tile_ptr[group_id];
        uint32_t stop  = B->tile_ptr[group_id+1];
        start = (start << 1) >> 1;
        stop  = (stop << 1) >> 1;

        if (start == stop)
            continue;

        for (uint32_t row_idx = start; row_idx <= stop; row_idx++) {
            if (B->row_ptr[row_idx] == B->row_ptr[row_idx+1]) {
                dirty = 1;
                break;
            }
        }

        if (dirty) {
            start |= sizeof(uint32_t) == 4
                                ? 0x80000000 : 0x8000000000000000;
            B->tile_ptr[group_id] = start;
        }
    }
    B->csr5_tail_tile_start = (B->tile_ptr[B->csr5_p-1] << 1) >> 1;

    // step 2. generate tile descriptor

    int bit_all_offset = B->csr5_bit_y_offset
                        + B->csr5_bit_scansum_offset;

    //generate_tile_descriptor_s1_kernel
    for (int par_id = 0; par_id < B->csr5_p-1; par_id++) {
        const ALPHA_INT row_start = B->tile_ptr[par_id]
                                        & 0x7FFFFFFF;
        const ALPHA_INT row_stop  = B->tile_ptr[par_id + 1]
                                        & 0x7FFFFFFF;

        for (int rid = row_start; rid <= row_stop; rid++) {
            int ptr = B->row_ptr[rid];
            int pid = ptr / (ALPHA_CSR5_OMEGA * B->csr5_sigma);

            if (pid == par_id) {
                int lx = (ptr / B->csr5_sigma) % ALPHA_CSR5_OMEGA;

                const int glid = ptr%B->csr5_sigma+bit_all_offset;
                const int ly = glid / 32;
                const int llid = glid % 32;

                const uint32_t val = 0x1 << (31 - llid);

                const int location = pid * ALPHA_CSR5_OMEGA
                    * B->csr5_num_packets
                    + ly * ALPHA_CSR5_OMEGA + lx;
                B->tile_desc[location] |= val;
            }
        }
    }

    //generate_tile_descriptor_s2_kernel
    int num_thread = 1; //omp_get_max_threads();
    ALPHA_INT *s_segn_scan_all, *s_present_all;

    s_segn_scan_all = alpha_memalign((uint64_t)(2 * ALPHA_CSR5_OMEGA * num_thread) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    s_present_all   = alpha_memalign((uint64_t)(2 * ALPHA_CSR5_OMEGA * num_thread) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);

    //int *s_segn_scan_all = (int *)malloc(2 * ALPHA_CSR5_OMEGA
    //                                   * sizeof(int) * num_thread);
    //int *s_present_all   = (int *)malloc(2 * ALPHA_CSR5_OMEGA
    //                                   * sizeof(int) * num_thread);
    for (ALPHA_INT i = 0; i < num_thread; i++)
        s_present_all[i * 2 * ALPHA_CSR5_OMEGA + ALPHA_CSR5_OMEGA]
            = 1;

    //const int bit_all_offset = bit_y_offset + bit_scansum_offset;

    //#pragma omp parallel for
    for (int par_id = 0; par_id < B->csr5_p-1; par_id++) {
        int tid = 0; //omp_get_thread_num();
        int *s_segn_scan = &s_segn_scan_all[tid * 2
                                            * ALPHA_CSR5_OMEGA];
        int *s_present = &s_present_all[tid * 2
                                            * ALPHA_CSR5_OMEGA];

        memset(s_segn_scan, 0, (ALPHA_CSR5_OMEGA + 1)*sizeof(int));
        memset(s_present, 0, ALPHA_CSR5_OMEGA * sizeof(int));

        bool with_empty_rows = (B->tile_ptr[par_id] >> 31) & 0x1;
        ALPHA_INT row_start       = B->tile_ptr[par_id]
                                        & 0x7FFFFFFF;
        const ALPHA_INT row_stop  = B->tile_ptr[par_id + 1]
                                        & 0x7FFFFFFF;

        if (row_start == row_stop)
            continue;

        //#pragma simd
        for (int lane_id = 0; lane_id < ALPHA_CSR5_OMEGA; lane_id++) {
            int start = 0, stop = 0, segn = 0;
            bool present = 0;
            uint32_t bitflag = 0;

            present |= !lane_id;

            // extract the first bit-flag packet
            int ly = 0;
            uint32_t first_packet = B->tile_desc[par_id
                * ALPHA_CSR5_OMEGA * B->csr5_num_packets+lane_id];
            bitflag = (first_packet << bit_all_offset)
                        | ((uint32_t)present << 31);
            start = !((bitflag >> 31) & 0x1);
            present |= (bitflag >> 31) & 0x1;

            for (int i = 1; i < B->csr5_sigma; i++) {
                if ((!ly && i == 32 - bit_all_offset)
                    || (ly && (i - (32 - bit_all_offset)) % 32==0))
                {
                    ly++;
                    bitflag = B->tile_desc[par_id
                            * ALPHA_CSR5_OMEGA
                            * B->csr5_num_packets
                            + ly * ALPHA_CSR5_OMEGA + lane_id];
                }
                const int norm_i = !ly ? i
                                    : i - (32 - bit_all_offset);
                stop += (bitflag >> (31 - norm_i % 32) ) & 0x1;
                present |= (bitflag >> (31 - norm_i % 32)) & 0x1;
            }

            // compute y_offset for all tiles
            segn = stop - start + present;
            segn = segn > 0 ? segn : 0;

            s_segn_scan[lane_id] = segn;

            // compute scansum_offset
            s_present[lane_id] = present;
        }

        //scan_single<int>(s_segn_scan, ALPHA_CSR5_OMEGA + 1);
        int old_val, new_val;
        old_val = s_segn_scan[0];
        s_segn_scan[0] = 0;
        for (int i = 1; i < ALPHA_CSR5_OMEGA + 1; i++) {
            new_val = s_segn_scan[i];
            s_segn_scan[i] = old_val + s_segn_scan[i-1];
            old_val = new_val;
        }

        if (with_empty_rows) {
            B->tile_desc_offset_ptr[par_id]
                = s_segn_scan[ALPHA_CSR5_OMEGA];
            B->tile_desc_offset_ptr[B->csr5_p] = 1;
        }

        //#pragma simd
        for (int lane_id = 0; lane_id < ALPHA_CSR5_OMEGA; lane_id++) {
            int y_offset = s_segn_scan[lane_id];

            int scansum_offset = 0;
            int next1 = lane_id + 1;
            if (s_present[lane_id]) {
                while ( ! s_present[next1] && next1 < ALPHA_CSR5_OMEGA)
                {
                    scansum_offset++;
                    next1++;
                }
            }

            uint32_t first_packet = B->tile_desc[par_id
                * ALPHA_CSR5_OMEGA * B->csr5_num_packets + lane_id];

            y_offset = lane_id ? y_offset - 1 : 0;

            first_packet |= y_offset << (32-B->csr5_bit_y_offset);
            first_packet |= scansum_offset << (32-bit_all_offset);

            B->tile_desc[par_id * ALPHA_CSR5_OMEGA
                * B->csr5_num_packets + lane_id] = first_packet;
        }
    }

    alpha_free(s_segn_scan_all);
    alpha_free(s_present_all);

    if (B->tile_desc_offset_ptr[B->csr5_p]) {
        //scan_single(B->tile_desc_offset_ptr, p+1);
        int old_val, new_val;
        old_val = B->tile_desc_offset_ptr[0];
        B->tile_desc_offset_ptr[0] = 0;
        for (int i = 1; i < B->csr5_p+1; i++)
        {
            new_val = B->tile_desc_offset_ptr[i];
            B->tile_desc_offset_ptr[i] = old_val
                                    + B->tile_desc_offset_ptr[i-1];
            old_val = new_val;
        }
    }

    B->csr5_num_offsets = B->tile_desc_offset_ptr[B->csr5_p];

    if (B->csr5_num_offsets) {
        B->tile_desc_offset = alpha_memalign((uint64_t)(B->csr5_num_offsets) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);

        //err = generate_tile_descriptor_offset
        const int bit_bitflag = 32 - bit_all_offset;

        //#pragma omp parallel for
        for (int par_id = 0; par_id < B->csr5_p-1; par_id++) {
            bool with_empty_rows = (B->tile_ptr[par_id] >> 31)&0x1;
            if (!with_empty_rows)
                continue;

            ALPHA_INT row_start       = B->tile_ptr[par_id]
                                            & 0x7FFFFFFF;
            const ALPHA_INT row_stop  = B->tile_ptr[par_id + 1]
                                            & 0x7FFFFFFF;

            int offset_pointer = B->tile_desc_offset_ptr[par_id];
            //#pragma simd
            for (int lane_id = 0; lane_id < ALPHA_CSR5_OMEGA; lane_id++) {
                bool local_bit;

                // extract the first bit-flag packet
                int ly = 0;
                uint32_t descriptor = B->tile_desc[par_id
                    * ALPHA_CSR5_OMEGA * B->csr5_num_packets
                    + lane_id];
                int y_offset = descriptor
                                >> (32 - B->csr5_bit_y_offset);

                descriptor = descriptor << bit_all_offset;
                descriptor = lane_id ? descriptor
                            : descriptor | 0x80000000;

                local_bit = (descriptor >> 31) & 0x1;

                if (local_bit && lane_id) {
                    const ALPHA_INT idx = par_id
                        * ALPHA_CSR5_OMEGA * B->csr5_sigma
                        + lane_id * B->csr5_sigma;
                    // binary search
                    ALPHA_INT start = 0;
                    ALPHA_INT stop = row_stop - row_start - 1;
                    ALPHA_INT median, key_median;
                    while (stop >= start) {
                        median = (stop + start) / 2;
                        key_median = B->row_ptr[row_start+1+median];
                        if (idx >= key_median)
                            start = median + 1;
                        else
                            stop = median - 1;
                    }
                    const ALPHA_INT y_index = start-1;

                    B->tile_desc_offset[offset_pointer + y_offset]
                        = y_index;

                    y_offset++;
                }

                for (int i = 1; i < B->csr5_sigma; i++) {
                    if ((!ly && i == bit_bitflag)
                        || (ly && !(31 & (i - bit_bitflag))))
                    {
                        ly++;
                        descriptor = B->tile_desc[par_id
                            * ALPHA_CSR5_OMEGA
                            * B->csr5_num_packets
                            + ly * ALPHA_CSR5_OMEGA + lane_id];
                    }
                    const int norm_i = 31 & (!ly
                                            ? i : i - bit_bitflag);

                    local_bit = (descriptor >> (31 - norm_i))&0x1;

                    if (local_bit) {
                        const ALPHA_INT idx = par_id
                            * ALPHA_CSR5_OMEGA * B->csr5_sigma
                            + lane_id * B->csr5_sigma + i;
                        // binary search
                        ALPHA_INT start = 0;
                        ALPHA_INT stop = row_stop-row_start-1;
                        ALPHA_INT median, key_median;
                        while (stop >= start) {
                            median = (stop + start) / 2;
                            key_median=B->row_ptr[row_start+1+median];
                            if (idx >= key_median)
                                start = median + 1;
                            else
                                stop = median - 1;
                        }
                        const ALPHA_INT y_index = start-1;

                        B->tile_desc_offset[offset_pointer
                                            + y_offset] = y_index;

                        y_offset++;
                    }
                }
            }
        }
    }

    // step 3. transpose column_index and value arrays
    //#pragma omp parallel for
    for (int par_id = 0; par_id < B->csr5_p; par_id++) {
        // if this is fast track tile, do not transpose it
        if (B->tile_ptr[par_id] == B->tile_ptr[par_id + 1]) {
            for (int idx = 0; idx < ALPHA_CSR5_OMEGA * B->csr5_sigma; idx++) {
                int src_idx = par_id * ALPHA_CSR5_OMEGA
                            * B->csr5_sigma + idx;
                B->col_idx[src_idx] = A->col_indx[src_idx];
                B->val[src_idx] = A->values[src_idx];
            }
            continue;
        }
        //#pragma simd
        if (par_id < B->csr5_p-1) {
            for (int idx = 0; idx < ALPHA_CSR5_OMEGA * B->csr5_sigma; idx++) {
                int idx_y = idx % B->csr5_sigma;
                int idx_x = idx / B->csr5_sigma;
                int src_idx = par_id * ALPHA_CSR5_OMEGA
                            * B->csr5_sigma + idx;
                int dst_idx = par_id * ALPHA_CSR5_OMEGA
                            * B->csr5_sigma + idx_y
                            * ALPHA_CSR5_OMEGA + idx_x;

                B->col_idx[dst_idx] = A->col_indx[src_idx];
                B->val[dst_idx] = A->values[src_idx];
            }
        }
        else { // the last tile
            for (int idx = par_id * ALPHA_CSR5_OMEGA * B->csr5_sigma; idx < B->nnz; idx++) {
                B->col_idx[idx] = A->col_indx[idx];
                B->val[idx] = A->values[idx];
            }
        }
    }

    //init deviece point
    B->d_col_idx    = NULL;
    B->d_row_ptr    = NULL;
    B->d_val        = NULL;

    B->d_tile_ptr             = NULL;
    B->d_tile_desc            = NULL;
    B->d_tile_desc_offset_ptr = NULL;
    B->d_tile_desc_offset     = NULL;
    B->d_calibrator           = NULL;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
