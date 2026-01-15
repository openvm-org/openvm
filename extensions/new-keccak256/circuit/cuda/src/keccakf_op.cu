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

// Main kernel for KeccakfOpChip trace generation
// Each thread processes one record (2 rows)
__global__ void keccakf_op_tracegen(
    Fp *d_trace,
    size_t height,
    uint32_t num_records,
    DeviceBufferConstView<KeccakfOpRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t record_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t records_to_fill = height / NUM_OP_ROWS_PER_INS;

    if (record_idx >= records_to_fill) {
        return;
    }

    // Each record produces 2 rows
    size_t row0_idx = record_idx * NUM_OP_ROWS_PER_INS;
    size_t row1_idx = record_idx * NUM_OP_ROWS_PER_INS + 1;

    RowSlice row0(d_trace + row0_idx, height);
    RowSlice row1(d_trace + row1_idx, height);

    // Initialize both rows to zero
    row0.fill_zero(0, sizeof(KeccakfOpCols<uint8_t>));
    row1.fill_zero(0, sizeof(KeccakfOpCols<uint8_t>));

    if (record_idx < num_records) {
        auto const &rec = d_records[record_idx];

        MemoryAuxColsFactory mem_helper(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr, bitwise_num_bits);

        // Compute postimage
        uint64_t state[25];
        memcpy(state, rec.preimage_buffer_bytes, KECCAK_WIDTH_BYTES);
        keccakf_permutation(state);
        uint8_t postimage_bytes[KECCAK_WIDTH_BYTES];
        memcpy(postimage_bytes, state, KECCAK_WIDTH_BYTES);

        // =========== Row 0: is_valid = 1, preimage buffer ===========
        {
            RowSlice row = row0;
            KECCAKF_OP_WRITE(pc, rec.pc);
            KECCAKF_OP_WRITE(is_valid, 1);
            KECCAKF_OP_WRITE(is_after_valid, 0);
            KECCAKF_OP_WRITE(timestamp, rec.timestamp);
            KECCAKF_OP_WRITE(rd_ptr, rec.rd_ptr);

            // Write buffer_ptr_limbs
            uint8_t buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS];
            buffer_ptr_limbs[0] = rec.buffer_ptr & 0xFF;
            buffer_ptr_limbs[1] = (rec.buffer_ptr >> 8) & 0xFF;
            buffer_ptr_limbs[2] = (rec.buffer_ptr >> 16) & 0xFF;
            buffer_ptr_limbs[3] = (rec.buffer_ptr >> 24) & 0xFF;
            KECCAKF_OP_WRITE_ARRAY(buffer_ptr_limbs, buffer_ptr_limbs);

            // Write preimage buffer
            KECCAKF_OP_WRITE_ARRAY(buffer, rec.preimage_buffer_bytes);

            // Fill rd_aux - memory read for rd_ptr
            uint32_t ts = rec.timestamp;
            mem_helper.fill(KECCAKF_OP_SLICE(rd_aux.base), rec.rd_aux.prev_timestamp, ts);
            ts++;

            // Fill buffer_word_aux - memory writes for 50 words
            for (uint32_t w = 0; w < KECCAK_WIDTH_WORDS; w++) {
                mem_helper.fill(
                    KECCAKF_OP_SLICE(buffer_word_aux[w]), rec.buffer_word_aux[w].prev_timestamp, ts
                );
                ts++;
            }

            // Range check for buffer pointer (scaled MSB limb)
            constexpr uint32_t RV32_TOTAL_BITS = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS;
            uint32_t scaled_limb = (buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1])
                                   << (RV32_TOTAL_BITS - pointer_max_bits);
            bitwise_lookup.add_range(scaled_limb, scaled_limb);

            // Range check for postimage bytes (pairs)
            for (size_t i = 0; i < KECCAK_WIDTH_BYTES; i += 2) {
                bitwise_lookup.add_range(postimage_bytes[i], postimage_bytes[i + 1]);
            }
        }

        // =========== Row 1: is_after_valid = 1, postimage buffer ===========
        {
            RowSlice row = row1;
            KECCAKF_OP_WRITE(pc, 0); // Not used in second row
            KECCAKF_OP_WRITE(is_valid, 0);
            KECCAKF_OP_WRITE(is_after_valid, 1);
            KECCAKF_OP_WRITE(timestamp, rec.timestamp); // Same timestamp
            KECCAKF_OP_WRITE(rd_ptr, 0);                // Not used in second row

            // buffer_ptr_limbs not needed in second row (zeroed)

            // Write postimage buffer
            KECCAKF_OP_WRITE_ARRAY(buffer, postimage_bytes);

            // Memory aux columns not used in second row (already zeroed)
        }
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
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(KeccakfOpCols<uint8_t>));

    uint32_t num_records = d_records.len();
    uint32_t records_to_fill = height / NUM_OP_ROWS_PER_INS;

    auto [grid, block] = kernel_launch_params(records_to_fill, 256);
    keccakf_op_tracegen<<<grid, block>>>(
        d_trace,
        height,
        num_records,
        d_records,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        pointer_max_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
