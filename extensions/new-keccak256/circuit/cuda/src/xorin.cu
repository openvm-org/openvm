#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "primitives/histogram.cuh"
#include "system/memory/controller.cuh"
#include "xorin/xorin.cuh"

using namespace xorin;

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
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        MemoryAuxColsFactory mem_helper(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr, bitwise_num_bits);

        // Fill instruction columns
        COL_WRITE_VALUE(row, XorinVmCols, instruction.pc, rec.from_pc);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.is_enabled, 1);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.buffer_ptr, rec.rd_ptr);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.input_ptr, rec.rs1_ptr);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.len_ptr, rec.rs2_ptr);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.buffer, rec.buffer);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.input, rec.input);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.len, rec.len);
        COL_WRITE_VALUE(row, XorinVmCols, instruction.start_timestamp, rec.timestamp);

        // Fill buffer limbs
        uint8_t buffer_bytes[4];
        memcpy(buffer_bytes, &rec.buffer, 4);
        COL_WRITE_ARRAY(row, XorinVmCols, instruction.buffer_limbs, buffer_bytes);

        // Fill input limbs
        uint8_t input_bytes[4];
        memcpy(input_bytes, &rec.input, 4);
        COL_WRITE_ARRAY(row, XorinVmCols, instruction.input_limbs, input_bytes);

        // Fill len limbs
        uint8_t len_bytes[4];
        memcpy(len_bytes, &rec.len, 4);
        COL_WRITE_ARRAY(row, XorinVmCols, instruction.len_limbs, len_bytes);

        uint32_t record_len = rec.len;
        uint32_t num_reads = (record_len + 3) / 4;  // div_ceil(len, 4)

        // Fill is_padding_bytes
        for (uint32_t i = 0; i < num_reads && i < XORIN_NUM_WORDS; i++) {
            COL_WRITE_VALUE(row, XorinVmCols, sponge.is_padding_bytes[i], 0);
        }
        for (uint32_t i = num_reads; i < XORIN_NUM_WORDS; i++) {
            COL_WRITE_VALUE(row, XorinVmCols, sponge.is_padding_bytes[i], 1);
        }

        // Fill sponge columns and request bitwise operations
        for (uint32_t i = 0; i < record_len && i < XORIN_RATE_BYTES; i++) {
            uint8_t buffer_byte = rec.buffer_limbs[i];
            uint8_t input_byte = rec.input_limbs[i];
            uint8_t result_byte = buffer_byte ^ input_byte;

            COL_WRITE_VALUE(row, XorinVmCols, sponge.preimage_buffer_bytes[i], buffer_byte);
            COL_WRITE_VALUE(row, XorinVmCols, sponge.input_bytes[i], input_byte);
            COL_WRITE_VALUE(row, XorinVmCols, sponge.postimage_buffer_bytes[i], result_byte);

            // Request bitwise XOR operation
            bitwise_lookup.add_xor(buffer_byte, input_byte);
        }

        // Zero-fill remaining sponge columns
        for (uint32_t i = record_len; i < XORIN_RATE_BYTES; i++) {
            COL_WRITE_VALUE(row, XorinVmCols, sponge.preimage_buffer_bytes[i], 0);
            COL_WRITE_VALUE(row, XorinVmCols, sponge.input_bytes[i], 0);
            COL_WRITE_VALUE(row, XorinVmCols, sponge.postimage_buffer_bytes[i], 0);
        }

        // Fill memory auxiliary columns
        // Timestamps follow the order from CPU trace: registers, buffer reads, input reads, buffer writes
        uint32_t timestamp = rec.timestamp;

        // Register aux cols (3 register reads)
        for (uint32_t t = 0; t < XORIN_REGISTER_READS; t++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(XorinVmCols, mem_oc.register_aux_cols[t].base)),
                rec.register_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }

        // Buffer bytes read aux cols
        for (uint32_t t = 0; t < num_reads && t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(XorinVmCols, mem_oc.buffer_bytes_read_aux_cols[t].base)),
                rec.buffer_read_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }
        // Zero remaining buffer read aux cols
        for (uint32_t t = num_reads; t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(XorinVmCols, mem_oc.buffer_bytes_read_aux_cols[t].base)));
        }

        // Input bytes read aux cols
        for (uint32_t t = 0; t < num_reads && t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(XorinVmCols, mem_oc.input_bytes_read_aux_cols[t].base)),
                rec.input_read_aux_cols[t].prev_timestamp,
                timestamp
            );
            timestamp++;
        }
        // Zero remaining input read aux cols
        for (uint32_t t = num_reads; t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(XorinVmCols, mem_oc.input_bytes_read_aux_cols[t].base)));
        }

        // Buffer bytes write aux cols
        for (uint32_t t = 0; t < num_reads && t < XORIN_NUM_WORDS; t++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(XorinVmCols, mem_oc.buffer_bytes_write_aux_cols[t].base)),
                rec.buffer_write_aux_cols[t].prev_timestamp,
                timestamp
            );
            // Fill prev_data for write aux cols
            COL_WRITE_ARRAY(row, XorinVmCols, mem_oc.buffer_bytes_write_aux_cols[t].prev_data, rec.buffer_write_aux_cols[t].prev_data);
            timestamp++;
        }
        // Zero remaining buffer write aux cols
        for (uint32_t t = num_reads; t < XORIN_NUM_WORDS; t++) {
            COL_FILL_ZERO(row, XorinVmCols, mem_oc.buffer_bytes_write_aux_cols[t]);
        }

        // Range check for pointer bounds
        uint32_t msl_rshift = 24;  // RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1) = 8 * 3
        uint32_t msl_lshift = 32 - pointer_max_bits;  // RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits

        bitwise_lookup.add_range(
            (rec.buffer >> msl_rshift) << msl_lshift,
            (rec.input >> msl_rshift) << msl_lshift
        );
        bitwise_lookup.add_range(
            (rec.len >> msl_rshift) << msl_lshift,
            (rec.len >> msl_rshift) << msl_lshift
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
