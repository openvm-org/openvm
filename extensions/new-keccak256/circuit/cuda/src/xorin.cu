#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "primitives/histogram.cuh"
#include "system/memory/controller.cuh"
#include "xorin/xorin.cuh"

using namespace xorin;
using namespace riscv;

#define XORIN_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, XorinVmCols, FIELD, VALUE)
#define XORIN_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, XorinVmCols, FIELD, VALUES)
#define XORIN_FILL_ZERO(FIELD) COL_FILL_ZERO(row, XorinVmCols, FIELD)
#define XORIN_SLICE(FIELD) row.slice_from(COL_INDEX(XorinVmCols, FIELD))

__global__ void xorin_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<XorinVmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
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

        MemoryAuxColsFactory mem_helper(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr, bitwise_num_bits);

        auto record_len = rec.len;
        // Clamp to max allowed length for safety
        if (record_len > XORIN_RATE_BYTES) {
            record_len = XORIN_RATE_BYTES;
        }
        auto num_reads = (record_len + 3) / 4;  // div_ceil(len, 4)

        // Fill instruction columns
        XORIN_WRITE(instruction.pc, rec.from_pc);
        XORIN_WRITE(instruction.is_enabled, 1);
        XORIN_WRITE(instruction.buffer_ptr, rec.rd_ptr);
        XORIN_WRITE(instruction.input_ptr, rec.rs1_ptr);
        XORIN_WRITE(instruction.len_ptr, rec.rs2_ptr);
        XORIN_WRITE(instruction.buffer, rec.buffer);
        XORIN_WRITE(instruction.input, rec.input);
        XORIN_WRITE(instruction.len, rec.len);
        XORIN_WRITE(instruction.start_timestamp, rec.timestamp);

        // Fill buffer/input/len limbs
        XORIN_WRITE_ARRAY(instruction.buffer_limbs, reinterpret_cast<const uint8_t*>(&rec.buffer));
        XORIN_WRITE_ARRAY(instruction.input_limbs, reinterpret_cast<const uint8_t*>(&rec.input));
        XORIN_WRITE_ARRAY(instruction.len_limbs, reinterpret_cast<const uint8_t*>(&rec.len));

        // Fill is_padding_bytes
        for (auto i = 0u; i < num_reads && i < XORIN_NUM_WORDS; i++) {
            XORIN_WRITE(sponge.is_padding_bytes[i], 0);
        }
        for (auto i = num_reads; i < XORIN_NUM_WORDS; i++) {
            XORIN_WRITE(sponge.is_padding_bytes[i], 1);
        }

        // Fill sponge columns and request bitwise operations
        for (auto i = 0u; i < record_len && i < XORIN_RATE_BYTES; i++) {
            XORIN_WRITE(sponge.preimage_buffer_bytes[i], rec.buffer_limbs[i]);
            XORIN_WRITE(sponge.input_bytes[i], rec.input_limbs[i]);
            XORIN_WRITE(sponge.postimage_buffer_bytes[i], rec.buffer_limbs[i] ^ rec.input_limbs[i]);
            bitwise_lookup.add_xor(rec.buffer_limbs[i], rec.input_limbs[i]);
        }

        // Zero-fill remaining sponge columns
        for (auto i = record_len; i < XORIN_RATE_BYTES; i++) {
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
        for (auto t = 0u; t < num_reads && t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill(
                XORIN_SLICE(mem_oc.buffer_bytes_read_aux_cols[t].base),
                rec.buffer_read_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }
        for (auto t = num_reads; t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill_zero(XORIN_SLICE(mem_oc.buffer_bytes_read_aux_cols[t].base));
        }

        // Input bytes read aux cols
        for (auto t = 0u; t < num_reads && t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill(
                XORIN_SLICE(mem_oc.input_bytes_read_aux_cols[t].base),
                rec.input_read_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }
        for (auto t = num_reads; t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill_zero(XORIN_SLICE(mem_oc.input_bytes_read_aux_cols[t].base));
        }

        // Buffer bytes write aux cols
        for (auto t = 0u; t < num_reads && t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill(
                XORIN_SLICE(mem_oc.buffer_bytes_write_aux_cols[t].base),
                rec.buffer_write_aux_cols[t].prev_timestamp,
                timestamp
            );
            XORIN_WRITE_ARRAY(mem_oc.buffer_bytes_write_aux_cols[t].prev_data, rec.buffer_write_aux_cols[t].prev_data);
            timestamp++;
        }
        for (auto t = num_reads; t < XORIN_NUM_WORDS; t++) {
            XORIN_FILL_ZERO(mem_oc.buffer_bytes_write_aux_cols[t]);
        }

        // Range check for pointer bounds
        constexpr uint32_t MSL_RSHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        constexpr uint32_t RV32_TOTAL_BITS = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS;
        bitwise_lookup.add_range(
            (rec.buffer >> MSL_RSHIFT) << (RV32_TOTAL_BITS - pointer_max_bits),
            (rec.input >> MSL_RSHIFT) << (RV32_TOTAL_BITS - pointer_max_bits)
        );
        bitwise_lookup.add_range(
            (rec.len >> MSL_RSHIFT) << (RV32_TOTAL_BITS - pointer_max_bits),
            (rec.len >> MSL_RSHIFT) << (RV32_TOTAL_BITS - pointer_max_bits)
        );

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
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(XorinVmCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    xorin_tracegen<<<grid, block>>>(
        d_trace,
        height,
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
