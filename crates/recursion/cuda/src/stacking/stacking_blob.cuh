#include "fpext.h"
#include <cstdint>

// Metadata (i.e. location in stacked matrices + dimensions) for a trace
struct StackedTraceData {
    uint32_t commit_idx;
    uint32_t start_col_idx;
    uint32_t start_row_idx;
    uint32_t log_height;
    uint32_t width;
    uint32_t need_rot;
};

// Metadata for a slice (i.e. trace column in stacked matrix)
struct StackedSliceData {
    uint32_t commit_idx;
    uint32_t col_idx;
    uint32_t row_idx;
    int32_t n;
    bool is_last_for_claim;
    uint32_t need_rot;
};

// Precomputation of eq * in, k_rot * in, and eq_bits
struct PolyPrecomputation {
    FpExt eq_in;
    FpExt k_rot_in;
    FpExt eq_bits;
};
