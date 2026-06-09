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
using openvm::U16_BITS;

// Computes the byte->cell base-pointer conversion carry and the per-block cell-offset add carries
// for one heap access group, registering the matching range-check counts. Mirrors the
// `compute_pointer_carries` closure in `main_chip/trace.rs`. Returns the base conversion carry
// (= `byte_hi & 1`) and writes one add carry per block (block `i`'s carry into the high cell limb)
// into `add_carry_out`.
static __device__ __forceinline__ uint32_t sha2_main_fill_pointer_carries(
    VariableRangeChecker &range_checker,
    uint32_t ptr,
    uint32_t hi_bits,
    uint32_t num_blocks,
    uint32_t cell_stride,
    uint32_t *add_carry_out
) {
    uint32_t byte_limbs[RV64_PTR_U16_LIMBS];
    ptr_to_u16_limbs(byte_limbs, ptr);
    uint32_t conv_carry = byte_limbs[1] & 1u;
    uint32_t base_cell_lo = (byte_limbs[0] + (conv_carry << U16_BITS)) >> 1;
    uint32_t base_cell_hi = byte_limbs[1] >> 1;
    range_checker.add_count(base_cell_hi, hi_bits);
    for (uint32_t i = 0; i < num_blocks; i++) {
        uint32_t sum_lo = base_cell_lo + i * cell_stride;
        range_checker.add_count(sum_lo & 0xffffu, U16_BITS);
        add_carry_out[i] = sum_lo >> U16_BITS;
    }
    return conv_carry;
}

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

    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);

    // Block cols
    SHA2_MAIN_WRITE_BLOCK(V, row, request_id, Fp(row_idx));
    {
        Fp message_u16s[V::BLOCK_U16S];
        bytes_to_u16_limbs(message_u16s, record.message_bytes);
        SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, message_u16s, message_u16s);

        Fp prev_state_u16s[V::STATE_U16S];
        Fp new_state_u16s[V::STATE_U16S];
        bytes_to_u16_limbs(prev_state_u16s, record.prev_state);
        bytes_to_u16_limbs(new_state_u16s, record.new_state);
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

    // Store low-32-bit register pointers as u16 cells.
    uint16_t dst_ptr_u16s[RV64_PTR_U16_LIMBS];
    uint16_t state_ptr_u16s[RV64_PTR_U16_LIMBS];
    uint16_t input_ptr_u16s[RV64_PTR_U16_LIMBS];
    ptr_to_u16_limbs(dst_ptr_u16s, header->dst_ptr);
    ptr_to_u16_limbs(state_ptr_u16s, header->state_ptr);
    ptr_to_u16_limbs(input_ptr_u16s, header->input_ptr);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, dst_ptr_limbs, dst_ptr_u16s);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, state_ptr_limbs, state_ptr_u16s);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, input_ptr_limbs, input_ptr_u16s);

    // Byte -> cell pointer conversion carries and per-block cell-offset carries, plus matching
    // range-check counts. Mirrors `compute_pointer_carries` in `main_chip/trace.rs`.
    uint32_t hi_bits = ptr_max_bits - U16_CELL_SIZE_BITS - U16_BITS;
    uint32_t read_cell_stride = SHA2_READ_SIZE / 2;
    uint32_t write_cell_stride = SHA2_WRITE_SIZE / 2;

    uint32_t input_add_carry[V::BLOCK_READS];
    uint32_t state_add_carry[V::STATE_READS];
    uint32_t write_add_carry[V::STATE_WRITES];
    uint32_t input_conv_carry = sha2_main_fill_pointer_carries(
        range_checker, header->input_ptr, hi_bits, V::BLOCK_READS, read_cell_stride, input_add_carry
    );
    uint32_t state_conv_carry = sha2_main_fill_pointer_carries(
        range_checker, header->state_ptr, hi_bits, V::STATE_READS, read_cell_stride, state_add_carry
    );
    uint32_t dst_conv_carry = sha2_main_fill_pointer_carries(
        range_checker, header->dst_ptr, hi_bits, V::STATE_WRITES, write_cell_stride, write_add_carry
    );

    SHA2_MAIN_WRITE_MEM(V, row, input_cell_carry, Fp(input_conv_carry));
    SHA2_MAIN_WRITE_MEM(V, row, state_cell_carry, Fp(state_conv_carry));
    SHA2_MAIN_WRITE_MEM(V, row, dst_cell_carry, Fp(dst_conv_carry));
    SHA2_MAIN_WRITE_ARRAY_MEM(V, row, input_add_carry, input_add_carry);
    SHA2_MAIN_WRITE_ARRAY_MEM(V, row, state_add_carry, state_add_carry);
    SHA2_MAIN_WRITE_ARRAY_MEM(V, row, write_add_carry, write_add_carry);

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
