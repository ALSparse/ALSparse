#pragma once

enum csrgemv_partition {
    SMALL_PARTITION,
    LARGE_PARTITION
};

#define MEMALIGN_FLAG() (1 << 1)
// #define ROW_PTR_FETCH_FLAG() (1 << 4)
