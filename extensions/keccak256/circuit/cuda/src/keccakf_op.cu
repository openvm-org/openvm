#include "fp.h"
#include "keccakf_op.cuh"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

using namespace keccakf_op;
using namespace riscv;

#define KECCAKF_OP_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, KeccakfOpCols, FIELD, VALUE)
#define KECCAKF_OP_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, KeccakfOpCols, FIELD, VALUES)
#define KECCAKF_OP_FILL_ZERO(FIELD) COL_FILL_ZERO(row, KeccakfOpCols, FIELD)
#define KECCAKF_OP_SLICE(FIELD) row.slice_from(COL_INDEX(KeccakfOpCols, FIELD))

// Fill the single trace row for one keccakf_op record.
//
// Marked __noinline__ so the large local working set (200 B keccak state union,
// MemoryAuxColsFactory, plus per-call spills around keccakf_permutation) lives in this
// helper's own stack frame instead of the kernel's. The kernel itself only does the
// index/dummy-row plumbing.
static __device__ __noinline__ void fill_keccakf_op_row(
    RowSlice row,
    KeccakfOpRecord const &rec,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    VariableRangeChecker range_checker(d_range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);

    // CUDA is little-endian, so the u64 word and byte representations below are the same memory
    // layout.
    union KeccakState {
        uint64_t u64[25];
        uint8_t bytes[KECCAK_WIDTH_BYTES];
    };
    // Compute postimage
    KeccakState state;
    memcpy(state.u64, rec.preimage_buffer_bytes, KECCAK_WIDTH_BYTES);
    keccakf_permutation(state.u64);

    KECCAKF_OP_WRITE(pc, rec.pc);
    KECCAKF_OP_WRITE(is_valid, 1);
    KECCAKF_OP_WRITE(timestamp, rec.timestamp);
    KECCAKF_OP_WRITE(rd_ptr, rec.rd_ptr);

    // Pack low 4 bytes of `buffer_ptr` (a u32 memory address) into 2 u16 cells. The
    // upper 4 bytes of the RV64 register are zero and hardcoded in the bus message.
    auto buffer_ptr_bytes = reinterpret_cast<uint8_t const *>(&rec.buffer_ptr);
    uint32_t buffer_ptr_limbs[BUFFER_PTR_NUM_LIMBS];
#pragma unroll
    for (size_t i = 0; i < BUFFER_PTR_NUM_LIMBS; i++) {
        buffer_ptr_limbs[i] =
            uint32_t(buffer_ptr_bytes[2 * i]) + 256u * uint32_t(buffer_ptr_bytes[2 * i + 1]);
    }
    KECCAKF_OP_WRITE_ARRAY(buffer_ptr_limbs, buffer_ptr_limbs);

    // Pack consecutive pairs of state bytes into u16 cells for preimage / postimage.
    uint16_t preimage_u16s[KECCAK_WIDTH_U16S];
    uint16_t postimage_u16s[KECCAK_WIDTH_U16S];
#pragma unroll
    for (size_t i = 0; i < KECCAK_WIDTH_U16S; i++) {
        preimage_u16s[i] = uint16_t(rec.preimage_buffer_bytes[2 * i]) |
                           (uint16_t(rec.preimage_buffer_bytes[2 * i + 1]) << 8);
        postimage_u16s[i] =
            uint16_t(state.bytes[2 * i]) | (uint16_t(state.bytes[2 * i + 1]) << 8);
    }
    KECCAKF_OP_WRITE_ARRAY(preimage, preimage_u16s);
    KECCAKF_OP_WRITE_ARRAY(postimage, postimage_u16s);

    // Fill rd_aux - memory read for rd_ptr
    uint32_t ts = rec.timestamp;
    mem_helper.fill(KECCAKF_OP_SLICE(rd_aux.base), rec.rd_aux.prev_timestamp, ts);
    ts++;

    // Fill buffer_word_aux - memory writes for 25 words
    for (uint32_t w = 0; w < KECCAK_WIDTH_MEM_OPS; w++) {
        mem_helper.fill(
            KECCAKF_OP_SLICE(buffer_word_aux[w]), rec.buffer_word_aux[w].prev_timestamp, ts
        );
        ts++;
    }

    // Range check for `buffer_ptr` (using pointer_max_bits): the high u16 cell of the
    // low 32 bits scaled by `1 << (32 - pointer_max_bits)` must fit in 16 bits.
    uint32_t buffer_ptr_msl_lshift = 32u - pointer_max_bits;
    uint32_t buffer_ptr_high_u16 = rec.buffer_ptr >> 16;
    range_checker.add_count(buffer_ptr_high_u16 << buffer_ptr_msl_lshift, 16);
}

// Main kernel for KeccakfOpChip trace generation
// Each thread processes one record (1 rows)
__global__ void keccakf_op_tracegen(
    Fp *d_trace,
    size_t height,
    uint32_t num_records,
    DeviceBufferConstView<KeccakfOpRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t record_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t records_to_fill = height / NUM_OP_ROWS_PER_INS;

    if (record_idx >= records_to_fill) {
        return;
    }

    // Each record produces 1 row
    size_t row0_idx = record_idx * NUM_OP_ROWS_PER_INS;

    RowSlice row0(d_trace + row0_idx, height);

    // Initialize the row to zero (also covers the dummy-row case below)
    row0.fill_zero(0, sizeof(KeccakfOpCols<uint8_t>));

    if (record_idx < num_records) {
        fill_keccakf_op_row(
            row0,
            d_records[record_idx],
            d_range_checker_ptr,
            range_checker_num_bins,
            pointer_max_bits,
            timestamp_max_bits
        );
    }
    // Dummy rows are already zeroed
}

#undef KECCAKF_OP_WRITE
#undef KECCAKF_OP_WRITE_ARRAY
#undef KECCAKF_OP_FILL_ZERO
#undef KECCAKF_OP_SLICE

extern "C" int _keccakf_op_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<KeccakfOpRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(KeccakfOpCols<uint8_t>));

    uint32_t num_records = d_records.len();
    uint32_t records_to_fill = height / NUM_OP_ROWS_PER_INS;

    auto [grid, block] = kernel_launch_params(records_to_fill, 256);
    keccakf_op_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        num_records,
        d_records,
        d_range_checker_ptr,
        range_checker_num_bins,
        pointer_max_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
