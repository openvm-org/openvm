#include "fp.h"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "xorin.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace xorin;
using namespace riscv;
using namespace keccak256;
using namespace program;
using openvm::U16_BITS;

#define XORIN_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, XorinVmCols, FIELD, VALUE)
#define XORIN_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, XorinVmCols, FIELD, VALUES)
#define XORIN_FILL_ZERO(FIELD) COL_FILL_ZERO(row, XorinVmCols, FIELD)
#define XORIN_SLICE(FIELD) row.slice_from(COL_INDEX(XorinVmCols, FIELD))

// Computes the byte->cell base-pointer conversion carry and the per-block cell-offset add carries
// for one heap access group, registering the matching range-check counts. Mirrors the CPU
// `fill_pointer_carries` closure in `xorin/trace.rs`. The per-block add carries (and their range
// checks) are computed for *every* block, padding or not, to match the AIR's `is_enabled`-gated
// `eval_add_const_u16_limbs`. The base high-limb range check is skipped when `register_base` is
// false (the buffer write group reuses the buffer read group's base conversion). Returns the base
// conversion carry.
__device__ __forceinline__ uint32_t xorin_fill_pointer_carries(
    VariableRangeChecker &range_checker,
    uint32_t ptr,
    uint32_t hi_bits,
    bool register_base,
    uint32_t add_carry_out[KECCAK_RATE_MEM_OPS]
) {
    uint32_t byte_limbs[RV64_PTR_U16_LIMBS];
    ptr_to_u16_limbs(byte_limbs, ptr);
    uint32_t conv_carry = byte_limbs[1] & 1u;
    uint32_t base_cell_lo = (byte_limbs[0] + (conv_carry << U16_BITS)) >> 1;
    uint32_t base_cell_hi = byte_limbs[1] >> 1;
    if (register_base) {
        range_checker.add_count(base_cell_hi, hi_bits);
    }
    uint32_t cell_stride = MEMORY_BLOCK_BYTES / U16_CELL_SIZE;
    for (uint32_t i = 0; i < KECCAK_RATE_MEM_OPS; i++) {
        uint32_t sum_lo = base_cell_lo + i * cell_stride;
        range_checker.add_count(sum_lo & 0xffffu, U16_BITS);
        add_carry_out[i] = sum_lo >> U16_BITS;
    }
    return conv_carry;
}

__global__ void xorin_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<XorinVmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        assert((rec.len % DEFAULT_BLOCK_SIZE) == 0);
        assert(rec.len <= XORIN_RATE_BYTES);
        assert((uint64_t)rec.buffer + rec.len <= (1ULL << pointer_max_bits));
        assert((uint64_t)rec.input + rec.len <= (1ULL << pointer_max_bits));
        assert(rec.len < (1ULL << pointer_max_bits));

        VariableRangeChecker range_checker(d_range_checker_ptr, range_checker_num_bins);
        MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr);

        auto record_len = rec.len;
        auto num_reads =
            (record_len + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;

        // Fill instruction columns
        XORIN_WRITE(instruction.pc, rec.from_pc);
        XORIN_WRITE(instruction.is_enabled, 1);
        XORIN_WRITE(instruction.buffer_reg_ptr, rec.rd_ptr);
        XORIN_WRITE(instruction.input_reg_ptr, rec.rs1_ptr);
        XORIN_WRITE(instruction.len_reg_ptr, rec.rs2_ptr);
        XORIN_WRITE(instruction.len, rec.len);
        XORIN_WRITE(instruction.start_timestamp, rec.timestamp);

        // Store low-32-bit pointers as u16 cells.
        uint16_t buffer_ptr_limbs[RV64_PTR_U16_LIMBS];
        uint16_t input_ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(buffer_ptr_limbs, rec.buffer);
        ptr_to_u16_limbs(input_ptr_limbs, rec.input);
        XORIN_WRITE_ARRAY(instruction.buffer_ptr_limbs, buffer_ptr_limbs);
        XORIN_WRITE_ARRAY(instruction.input_ptr_limbs, input_ptr_limbs);
        XORIN_WRITE(instruction.len_limb, static_cast<uint8_t>(rec.len));

        // Fill is_padding_bytes
        for (auto i = 0u; i < num_reads && i < KECCAK_RATE_MEM_OPS; i++) {
            XORIN_WRITE(sponge.is_padding_bytes[i], 0);
        }
        for (auto i = num_reads; i < KECCAK_RATE_MEM_OPS; i++) {
            XORIN_WRITE(sponge.is_padding_bytes[i], 1);
        }

        // Bytes covered by active 8-byte memory blocks.
        auto bytes_covered = num_reads * MEMORY_BLOCK_BYTES;

        // Fill sponge columns and request bitwise operations for the xor'd region.
        for (auto i = 0u; i < record_len && i < XORIN_RATE_BYTES; i++) {
            XORIN_WRITE(sponge.preimage_buffer_bytes[i], rec.buffer_limbs[i]);
            XORIN_WRITE(sponge.input_bytes[i], rec.input_limbs[i]);
            XORIN_WRITE(sponge.postimage_buffer_bytes[i], rec.buffer_limbs[i] ^ rec.input_limbs[i]);
            bitwise_lookup.add_xor(rec.buffer_limbs[i], rec.input_limbs[i]);
        }

        // Zero-fill the remaining padding sponge columns.
        for (auto i = bytes_covered; i < XORIN_RATE_BYTES; i++) {
            XORIN_WRITE(sponge.preimage_buffer_bytes[i], Fp::zero());
            XORIN_WRITE(sponge.input_bytes[i], Fp::zero());
            XORIN_WRITE(sponge.postimage_buffer_bytes[i], Fp::zero());
        }

        // Timestamps follow the order from CPU trace: registers, buffer reads, input reads, buffer writes
        auto timestamp = rec.timestamp;

// Register aux cols (3 register reads)
#pragma unroll
        for (auto t = 0u; t < XORIN_REGISTER_READS; t++) {
            mem_helper.fill(
                XORIN_SLICE(mem_oc.register_aux_cols[t].base),
                rec.register_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }

        // Buffer bytes read aux cols
        for (auto t = 0u; t < num_reads && t < KECCAK_RATE_MEM_OPS; t++) {
            mem_helper.fill(
                XORIN_SLICE(mem_oc.buffer_bytes_read_aux_cols[t].base),
                rec.buffer_read_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }
        for (auto t = num_reads; t < KECCAK_RATE_MEM_OPS; t++) {
            mem_helper.fill_zero(XORIN_SLICE(mem_oc.buffer_bytes_read_aux_cols[t].base));
        }

        // Input bytes read aux cols
        for (auto t = 0u; t < num_reads && t < KECCAK_RATE_MEM_OPS; t++) {
            mem_helper.fill(
                XORIN_SLICE(mem_oc.input_bytes_read_aux_cols[t].base),
                rec.input_read_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }
        for (auto t = num_reads; t < KECCAK_RATE_MEM_OPS; t++) {
            mem_helper.fill_zero(XORIN_SLICE(mem_oc.input_bytes_read_aux_cols[t].base));
        }

        // Buffer bytes write aux cols
        for (auto t = 0u; t < num_reads && t < KECCAK_RATE_MEM_OPS; t++) {
            mem_helper.fill(
                XORIN_SLICE(mem_oc.buffer_bytes_write_aux_cols[t].base),
                rec.buffer_write_aux_cols[t].prev_timestamp,
                timestamp
            );
            Fp packed_prev[BLOCK_FE_WIDTH];
            pack_u8_block_bytes(packed_prev, rec.buffer_write_aux_cols[t].prev_data);
            XORIN_WRITE_ARRAY(
                mem_oc.buffer_bytes_write_aux_cols[t].prev_data,
                packed_prev
            );
            timestamp++;
        }
        for (auto t = num_reads; t < KECCAK_RATE_MEM_OPS; t++) {
            XORIN_FILL_ZERO(mem_oc.buffer_bytes_write_aux_cols[t]);
        }

        // Byte -> cell pointer conversion carries and per-block cell-offset carries, plus matching
        // range-check counts. Mirrors `xorin/trace.rs`.
        uint32_t hi_bits = uint32_t(pointer_max_bits) - U16_CELL_SIZE_BITS - U16_BITS;

        uint32_t buffer_add_carry[KECCAK_RATE_MEM_OPS];
        uint32_t buffer_conv_carry =
            xorin_fill_pointer_carries(range_checker, rec.buffer, hi_bits, true, buffer_add_carry);
        XORIN_WRITE(mem_oc.buffer_cell_carry, buffer_conv_carry);
        XORIN_WRITE_ARRAY(mem_oc.buffer_read_add_carry, buffer_add_carry);

        uint32_t input_add_carry[KECCAK_RATE_MEM_OPS];
        uint32_t input_conv_carry =
            xorin_fill_pointer_carries(range_checker, rec.input, hi_bits, true, input_add_carry);
        XORIN_WRITE(mem_oc.input_cell_carry, input_conv_carry);
        XORIN_WRITE_ARRAY(mem_oc.input_read_add_carry, input_add_carry);

        // The write reuses the converted `buffer` base cell pointer; only the per-block write add
        // carries (and their range checks) are needed. The base conversion was registered above.
        uint32_t buffer_write_add_carry[KECCAK_RATE_MEM_OPS];
        xorin_fill_pointer_carries(range_checker, rec.buffer, hi_bits, false, buffer_write_add_carry);
        XORIN_WRITE_ARRAY(mem_oc.buffer_write_add_carry, buffer_write_add_carry);

    } else {
        // Zero-fill padding rows
        row.fill_zero(0, sizeof(XorinVmCols<uint8_t>));
    }
}

extern "C" int _xorin_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<XorinVmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(XorinVmCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height, 512);

    xorin_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        pointer_max_bits,
        timestamp_max_bits
    );

    return CHECK_KERNEL();
}
