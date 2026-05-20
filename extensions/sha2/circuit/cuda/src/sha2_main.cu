#include "block_hasher/variant.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "main/columns.cuh"
#include "main/record.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace riscv;
using namespace sha2;

// Body shared by both the inlined (SHA-256) and outlined (SHA-512) paths.
//
// Marked `__forceinline__` so it always inlines into its caller; this lets us
// share one definition while still controlling whether the work ends up in the
// kernel's frame (SHA-256) or in a separate function frame (SHA-512).
template <typename V>
static __device__ __forceinline__ void sha2_main_row_body(
    RowSlice row,
    uint32_t row_idx,
    Sha2RecordMut<V> record,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    Sha2RecordHeader<V> *header = record.header;

    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
    );

    // Block cols.
    // `message_u16s`, `prev_state`, `new_state` are all u16-shaped: pack each pair of
    // consecutive bytes into one cell.
    SHA2_MAIN_WRITE_BLOCK(V, row, request_id, Fp(row_idx));
    {
        Fp message_u16s[V::BLOCK_U16S];
#pragma unroll
        for (size_t k = 0; k < V::BLOCK_U16S; k++) {
            message_u16s[k] = Fp(
                static_cast<uint32_t>(record.message_bytes[2 * k])
                | (static_cast<uint32_t>(record.message_bytes[2 * k + 1]) << 8)
            );
        }
        SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, message_u16s, message_u16s);

        Fp prev_state_u16s[V::STATE_U16S];
        Fp new_state_u16s[V::STATE_U16S];
#pragma unroll
        for (size_t k = 0; k < V::STATE_U16S; k++) {
            prev_state_u16s[k] = Fp(
                static_cast<uint32_t>(record.prev_state[2 * k])
                | (static_cast<uint32_t>(record.prev_state[2 * k + 1]) << 8)
            );
            new_state_u16s[k] = Fp(
                static_cast<uint32_t>(record.new_state[2 * k])
                | (static_cast<uint32_t>(record.new_state[2 * k + 1]) << 8)
            );
        }
        SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, prev_state, prev_state_u16s);
        SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, new_state, new_state_u16s);
    }

    // Instruction cols
    SHA2_MAIN_WRITE_INSTR(V, row, is_enabled, Fp::one());
    SHA2_MAIN_WRITE_INSTR(V, row, from_state.timestamp, header->timestamp);
    SHA2_MAIN_WRITE_INSTR(V, row, from_state.pc, header->from_pc);
    SHA2_MAIN_WRITE_INSTR(V, row, dst_reg_ptr, header->dst_reg_ptr);
    SHA2_MAIN_WRITE_INSTR(V, row, state_reg_ptr, header->state_reg_ptr);
    SHA2_MAIN_WRITE_INSTR(V, row, input_reg_ptr, header->input_reg_ptr);

    // Pack the low 32 bits of each register pointer into 2 u16 cells.
    Fp dst_ptr_u16s[RV64_PTR_U16_LIMBS];
    Fp state_ptr_u16s[RV64_PTR_U16_LIMBS];
    Fp input_ptr_u16s[RV64_PTR_U16_LIMBS];
    u32_to_le_u16_limbs(dst_ptr_u16s, header->dst_ptr);
    u32_to_le_u16_limbs(state_ptr_u16s, header->state_ptr);
    u32_to_le_u16_limbs(input_ptr_u16s, header->input_ptr);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, dst_ptr_limbs, dst_ptr_u16s);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, state_ptr_limbs, state_ptr_u16s);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, input_ptr_limbs, input_ptr_u16s);

    // Range-check the high u16 of each pointer via the variable range checker.
    uint32_t shift = 1u << (RV64_U16_LIMB_BITS * RV64_PTR_U16_LIMBS - ptr_max_bits);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    uint32_t hi_dst = static_cast<uint32_t>(dst_ptr_u16s[RV64_PTR_U16_LIMBS - 1].asUInt32());
    uint32_t hi_state = static_cast<uint32_t>(state_ptr_u16s[RV64_PTR_U16_LIMBS - 1].asUInt32());
    uint32_t hi_input = static_cast<uint32_t>(input_ptr_u16s[RV64_PTR_U16_LIMBS - 1].asUInt32());
    range_checker.add_count(hi_dst * shift, RV64_U16_LIMB_BITS);
    range_checker.add_count(hi_state * shift, RV64_U16_LIMB_BITS);
    range_checker.add_count(hi_input * shift, RV64_U16_LIMB_BITS);

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
        Fp packed_prev[BLOCK_FE_WIDTH];
        pack_u8_block_bytes(packed_prev, record.write_aux[i].prev_data);
        write_aux.write_array(
            COL_INDEX(MemoryWriteAuxCols, prev_data),
            BLOCK_FE_WIDTH,
            packed_prev
        );
        RowSlice base_slice = write_aux.slice_from(COL_INDEX(MemoryWriteAuxCols, base));
        mem_helper.fill(base_slice, record.write_aux[i].prev_timestamp, timestamp);
        timestamp += 1;
    }
}

// __noinline__ wrapper used by the SHA-512 instantiation. SHA-512 has twice as
// many memory aux iterations as SHA-256 (BLOCK_READS=32, STATE_READS=16,
// STATE_WRITES=16), so outlining keeps the heavy working set in its own stack
// frame rather than bloating the kernel's.
template <typename V>
static __device__ __noinline__ void sha2_main_row_outlined(
    RowSlice row,
    uint32_t row_idx,
    Sha2RecordMut<V> record,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    sha2_main_row_body<V>(
        row,
        row_idx,
        record,
        ptr_max_bits,
        range_checker_ptr,
        range_checker_num_bins,
        timestamp_max_bits
    );
}

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
    // SHA-512 routes through the __noinline__ wrapper to keep the heavy
    // working set out of the kernel's stack frame; SHA-256 inlines directly
    // since it fits comfortably in registers.
    if constexpr (V::WORD_BITS > 32) {
        sha2_main_row_outlined<V>(
            row,
            row_idx,
            record,
            ptr_max_bits,
            range_checker_ptr,
            range_checker_num_bins,
            timestamp_max_bits
        );
    } else {
        sha2_main_row_body<V>(
            row,
            row_idx,
            record,
            ptr_max_bits,
            range_checker_ptr,
            range_checker_num_bins,
            timestamp_max_bits
        );
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
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    auto [grid_size, block_size] = kernel_launch_params(trace_height, 256);
    sha2_main_tracegen<V><<<grid_size, block_size, 0, stream>>>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
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
    uint32_t timestamp_max_bits,
    cudaStream_t stream
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
        timestamp_max_bits,
        stream
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
    uint32_t timestamp_max_bits,
    cudaStream_t stream
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
        timestamp_max_bits,
        stream
    );
}
}
