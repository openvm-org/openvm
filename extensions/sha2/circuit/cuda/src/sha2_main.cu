#include "block_hasher/variant.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "main/columns.cuh"
#include "main/record.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace riscv;
using namespace sha2;

// ===== MAIN CHIP KERNEL =====
template <typename V>
__global__ void sha2_main_tracegen(
    Fp *trace,
    size_t trace_height,
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= trace_height) {
        return;
    }

    RowSlice row(trace + row_idx, trace_height);
    row.fill_zero(0, sha2::Sha2MainLayout<V>::WIDTH);

    if (row_idx >= num_records) {
        return;
    }

    Sha2RecordMut<V> record(records + record_offsets[row_idx]);
    Sha2RecordHeader<V> *header = record.header;

    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);
    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
    );

    // Block cols
    SHA2_MAIN_WRITE_BLOCK(V, row, request_id, Fp(row_idx));
    SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, message_bytes, record.message_bytes);
    SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, prev_state, record.prev_state);
    SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, new_state, record.new_state);

    // Instruction cols
    SHA2_MAIN_WRITE_INSTR(V, row, is_enabled, Fp::one());
    SHA2_MAIN_WRITE_INSTR(V, row, from_state.timestamp, header->timestamp);
    SHA2_MAIN_WRITE_INSTR(V, row, from_state.pc, header->from_pc);
    SHA2_MAIN_WRITE_INSTR(V, row, dst_reg_ptr, header->dst_reg_ptr);
    SHA2_MAIN_WRITE_INSTR(V, row, state_reg_ptr, header->state_reg_ptr);
    SHA2_MAIN_WRITE_INSTR(V, row, input_reg_ptr, header->input_reg_ptr);

    uint8_t dst_ptr_bytes[RV32_REGISTER_NUM_LIMBS];
    uint8_t state_ptr_bytes[RV32_REGISTER_NUM_LIMBS];
    uint8_t input_ptr_bytes[RV32_REGISTER_NUM_LIMBS];
    memcpy(dst_ptr_bytes, &header->dst_ptr, sizeof(uint32_t));
    memcpy(state_ptr_bytes, &header->state_ptr, sizeof(uint32_t));
    memcpy(input_ptr_bytes, &header->input_ptr, sizeof(uint32_t));

    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, dst_ptr_limbs, dst_ptr_bytes);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, state_ptr_limbs, state_ptr_bytes);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, input_ptr_limbs, input_ptr_bytes);

    // Range checks on top limbs
    uint8_t needs_range_check[4] = {
        dst_ptr_bytes[RV32_REGISTER_NUM_LIMBS - 1],
        state_ptr_bytes[RV32_REGISTER_NUM_LIMBS - 1],
        input_ptr_bytes[RV32_REGISTER_NUM_LIMBS - 1],
        input_ptr_bytes[RV32_REGISTER_NUM_LIMBS - 1],
    };
    uint32_t shift = 1u << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - ptr_max_bits);
    for (int i = 0; i < 4; i += 2) {
        bitwise_lookup.add_range(
            static_cast<uint32_t>(needs_range_check[i]) * shift,
            static_cast<uint32_t>(needs_range_check[i + 1]) * shift
        );
    }

    // Memory aux
    uint32_t timestamp = header->timestamp;
    for (int i = 0; i < static_cast<int>(sha2::SHA2_REGISTER_READS); i++) {
        RowSlice reg_aux = SHA2_MAIN_SLICE_MEM(V, row, register_aux[i]);
        RowSlice base_slice = reg_aux.slice_from(COL_INDEX(MemoryReadAuxCols, base));
        mem_helper.fill(base_slice, header->register_reads_aux[i].prev_timestamp, timestamp);
        timestamp += 1;
    }

    for (int i = 0; i < static_cast<int>(V::BLOCK_READS); i++) {
        RowSlice input_aux = SHA2_MAIN_SLICE_MEM(V, row, input_reads[i]);
        RowSlice base_slice = input_aux.slice_from(COL_INDEX(MemoryReadAuxCols, base));
        mem_helper.fill(base_slice, record.input_reads_aux[i].prev_timestamp, timestamp);
        timestamp += 1;
    }

    for (int i = 0; i < static_cast<int>(V::STATE_READS); i++) {
        RowSlice state_aux = SHA2_MAIN_SLICE_MEM(V, row, state_reads[i]);
        RowSlice base_slice = state_aux.slice_from(COL_INDEX(MemoryReadAuxCols, base));
        mem_helper.fill(base_slice, record.state_reads_aux[i].prev_timestamp, timestamp);
        timestamp += 1;
    }

    for (int i = 0; i < static_cast<int>(V::STATE_WRITES); i++) {
        RowSlice write_aux = SHA2_MAIN_SLICE_MEM(V, row, write_aux[i]);
        write_aux.write_array(
            COL_INDEX(MemoryWriteAuxCols, prev_data),
            sha2::SHA2_WRITE_SIZE,
            record.write_aux[i].prev_data
        );
        RowSlice base_slice = write_aux.slice_from(COL_INDEX(MemoryWriteAuxCols, base));
        mem_helper.fill(base_slice, record.write_aux[i].prev_timestamp, timestamp);
        timestamp += 1;
    }
}

// ===== HOST LAUNCHER FUNCTIONS =====

template <typename V>
int launch_sha2_main_tracegen(
    Fp *d_trace,
    size_t trace_height,
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    auto [grid_size, block_size] = kernel_launch_params(trace_height, 256);
    sha2_main_tracegen<V><<<grid_size, block_size>>>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// Explicit instantiations for SHA-256 and SHA-512
extern "C" {
int launch_sha256_main_tracegen(
    Fp *d_trace,
    size_t trace_height,
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    return launch_sha2_main_tracegen<Sha256Variant>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
}

int launch_sha512_main_tracegen(
    Fp *d_trace,
    size_t trace_height,
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    return launch_sha2_main_tracegen<Sha512Variant>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
}
}
