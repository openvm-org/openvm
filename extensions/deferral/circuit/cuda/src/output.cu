#include <cassert>
#include <cstddef>
#include <cstdint>

#include "canonicity.cuh"
#include "def_poseidon2_buffer.cuh"
#include "def_types.h"
#include "fp.h"
#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/fp_array.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace deferral;
using namespace canonicity;
using namespace lookup;

struct DeferralOutputRecordHeader {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs_ptr;
    uint32_t deferral_idx;
    uint32_t num_rows;

    uint8_t rd_val[RV64_WORD_NUM_LIMBS];
    uint8_t rs_val[RV64_WORD_NUM_LIMBS];
    MemoryReadAuxRecord rd_aux;
    MemoryReadAuxRecord rs_aux;

    MemoryReadAuxRecord output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];
};

struct DeferralOutputPerCall {
    uint8_t output_commit[COMMIT_NUM_BYTES];
};

struct DeferralOutputPerRow {
    uint32_t header_offset;
    uint32_t section_idx;
    uint32_t call_idx;
    Fp poseidon2_res[DIGEST_SIZE];
};

template <typename T> using MemoryWriteAuxColsDef = MemoryWriteAuxCols<T, MEMORY_OP_SIZE>;

__device__ __forceinline__ size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ void write_canonicity_aux(
    RowSlice row,
    size_t base_offset,
    size_t aux_idx,
    const CanonicityAuxCols<Fp> &aux
) {
    constexpr size_t aux_stride = sizeof(CanonicityAuxCols<uint8_t>);
    RowSlice aux_row = row.slice_from(base_offset + aux_idx * aux_stride);
    COL_WRITE_ARRAY(aux_row, CanonicityAuxCols, diff_marker, aux.diff_marker);
    COL_WRITE_VALUE(aux_row, CanonicityAuxCols, diff_val, aux.diff_val);
}

template <typename T> struct DeferralOutputCols {
    // Indicates the status of this row, i.e. if it is valid and where it is in a
    // section of rows that correspond to a single opcode invocation
    T is_valid;
    T is_first;
    T is_last;
    T section_idx;

    // Initial execution state + instruction operands
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs_ptr;
    T deferral_idx;

    // Heap pointers + auxiliary read columns
    T rd_val[RV64_WORD_NUM_LIMBS];
    T rs_val[RV64_WORD_NUM_LIMBS];
    MemoryReadAuxCols<T> rd_aux;
    MemoryReadAuxCols<T> rs_aux;

    // Read data and auxiliary columns. output_commit and output_len are read
    // contiguously from heap with layout [output_commit || output_len]. The
    // onion hash of all bytes written by this opcode invocation is constrained
    // to output_commit.
    T output_commit[COMMIT_NUM_BYTES];
    T output_len[F_NUM_BYTES];
    MemoryReadAuxCols<T> output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];

    // Auxiliary columns to ensure the canonicity of each F byte decomposition in
    // output_commit.
    CanonicityAuxCols<T> output_commit_lt_aux[DIGEST_SIZE];

    // Initial [def_idx, output_len, 0, ...] digest on the first row; on non-first
    // rows bytes raw_output[local_idx * DIGEST_SIZE..(local_idx + 1) * DIGEST_SIZE]
    // written to memory and auxiliary columns.
    T sponge_inputs[DIGEST_SIZE];
    MemoryWriteAuxCols<T, MEMORY_OP_SIZE> write_bytes_aux[DIGEST_MEMORY_OPS];

    // Capacity of the permutation of write_bytes and the previous row's capacity on
    // non-last rows, compression on the last row.
    T poseidon2_res[DIGEST_SIZE];
};

__global__ void deferral_output_tracegen(
    Fp *trace,
    size_t height,
    const uint8_t *raw_records,
    const DeferralOutputPerCall *per_call,
    const DeferralOutputPerRow *per_row,
    const size_t num_valid,
    uint32_t *count_ptr,
    const size_t num_def_circuits,
    uint32_t *range_checker_ptr,
    const uint32_t range_checker_num_bins,
    const uint32_t timestamp_max_bits,
    uint32_t *bitwise_ptr,
    const size_t address_bits,
    FpArray<16> *poseidon2_records,
    DeferralPoseidon2Count *poseidon2_counts,
    uint32_t *poseidon2_idx,
    const size_t poseidon2_capacity
) {
    const uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_valid) {
        row.fill_zero(0, sizeof(DeferralOutputCols<uint8_t>));
        return;
    }

    const DeferralOutputPerRow row_data = per_row[row_idx];
    const uint32_t section_idx = row_data.section_idx;
    const DeferralOutputPerCall call_data = per_call[row_data.call_idx];

    const uint8_t *record_start = raw_records + row_data.header_offset;
    const DeferralOutputRecordHeader header =
        *reinterpret_cast<const DeferralOutputRecordHeader *>(record_start);

    const uint32_t output_len = (header.num_rows - 1) * DIGEST_SIZE;
    const bool is_first = section_idx == 0;
    const bool is_last = section_idx + 1 == header.num_rows;

    Histogram count_buffer(count_ptr, num_def_circuits);
    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
    );
    BitwiseOperationLookup bitwise_buffer(bitwise_ptr, RV64_CELL_BITS);
    DeferralPoseidon2Buffer poseidon2_buffer(
        poseidon2_records, poseidon2_counts, poseidon2_idx, poseidon2_capacity
    );

    COL_WRITE_VALUE(row, DeferralOutputCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, DeferralOutputCols, is_first, is_first ? Fp::one() : Fp::zero());
    COL_WRITE_VALUE(row, DeferralOutputCols, is_last, is_last ? Fp::one() : Fp::zero());
    COL_WRITE_VALUE(row, DeferralOutputCols, section_idx, section_idx);

    COL_WRITE_VALUE(row, DeferralOutputCols, from_state.pc, header.from_pc);
    COL_WRITE_VALUE(row, DeferralOutputCols, from_state.timestamp, header.from_timestamp);
    COL_WRITE_VALUE(row, DeferralOutputCols, rd_ptr, header.rd_ptr);
    COL_WRITE_VALUE(row, DeferralOutputCols, rs_ptr, header.rs_ptr);
    COL_WRITE_VALUE(row, DeferralOutputCols, deferral_idx, header.deferral_idx);

    COL_WRITE_ARRAY(row, DeferralOutputCols, rd_val, header.rd_val);
    COL_WRITE_ARRAY(row, DeferralOutputCols, rs_val, header.rs_val);

    COL_WRITE_ARRAY(row, DeferralOutputCols, output_commit, call_data.output_commit);
    const uint8_t output_len_bytes[F_NUM_BYTES] = {
        static_cast<uint8_t>(output_len & 0xffu),
        static_cast<uint8_t>((output_len >> 8) & 0xffu),
        static_cast<uint8_t>((output_len >> 16) & 0xffu),
        static_cast<uint8_t>((output_len >> 24) & 0xffu),
    };
    COL_WRITE_ARRAY(row, DeferralOutputCols, output_len, output_len_bytes);

    if (is_first) {
        count_buffer.add_count(header.deferral_idx);

        const uint32_t limb_shift_bits = RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - address_bits;
        bitwise_buffer.add_range(
            static_cast<uint32_t>(header.rd_val[RV64_WORD_NUM_LIMBS - 1]) << limb_shift_bits,
            static_cast<uint32_t>(header.rs_val[RV64_WORD_NUM_LIMBS - 1]) << limb_shift_bits
        );
        bitwise_buffer.add_range(
            static_cast<uint32_t>(output_len_bytes[RV64_WORD_NUM_LIMBS - 1]) << limb_shift_bits,
            0
        );

        mem_helper.fill(
            row.slice_from(COL_INDEX(DeferralOutputCols, rd_aux)),
            header.rd_aux.prev_timestamp,
            header.from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(DeferralOutputCols, rs_aux)),
            header.rs_aux.prev_timestamp,
            header.from_timestamp + 1
        );
        constexpr size_t read_aux_stride = sizeof(MemoryReadAuxCols<uint8_t>);
#pragma unroll
        for (size_t chunk_idx = 0; chunk_idx < OUTPUT_TOTAL_MEMORY_OPS; ++chunk_idx) {
            mem_helper.fill(
                row.slice_from(
                    COL_INDEX(DeferralOutputCols, output_commit_and_len_aux) +
                    chunk_idx * read_aux_stride
                ),
                header.output_commit_and_len_aux[chunk_idx].prev_timestamp,
                header.from_timestamp + 2 + static_cast<uint32_t>(chunk_idx)
            );
        }

        uint32_t output_commit_rcs[DIGEST_SIZE];
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            CanonicityAuxCols<Fp> aux;
            Fp x_le[F_NUM_BYTES];
#pragma unroll
            for (size_t j = 0; j < F_NUM_BYTES; ++j) {
                x_le[j] = Fp(call_data.output_commit[i * F_NUM_BYTES + j]);
            }
            output_commit_rcs[i] = generate_subrow(x_le, aux);
            write_canonicity_aux(row, COL_INDEX(DeferralOutputCols, output_commit_lt_aux), i, aux);
        }
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; i += 2) {
            bitwise_buffer.add_range(output_commit_rcs[i], output_commit_rcs[i + 1]);
        }

        Fp sponge_inputs[DIGEST_SIZE];
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            sponge_inputs[i] = Fp::zero();
        }
        sponge_inputs[0] = Fp(header.deferral_idx);
        sponge_inputs[1] = Fp(output_len);
        COL_WRITE_ARRAY(row, DeferralOutputCols, sponge_inputs, sponge_inputs);

        COL_FILL_ZERO(row, DeferralOutputCols, write_bytes_aux);
    } else {
        const uint8_t *header_end = record_start + sizeof(DeferralOutputRecordHeader);
        const uint8_t *write_bytes_start = header_end + (section_idx - 1) * DIGEST_SIZE;
        const size_t write_aux_offset = align_up(
            sizeof(DeferralOutputRecordHeader) + output_len,
            alignof(MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE>)
        );
        const auto *write_aux = reinterpret_cast<const MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE> *>(
            record_start + write_aux_offset
        );

        COL_FILL_ZERO(row, DeferralOutputCols, rd_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, rs_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, output_commit_and_len_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, output_commit_lt_aux);

        COL_WRITE_ARRAY(row, DeferralOutputCols, sponge_inputs, write_bytes_start);

#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; i += 2) {
            bitwise_buffer.add_range(write_bytes_start[i], write_bytes_start[i + 1]);
        }

        constexpr size_t write_aux_stride = sizeof(MemoryWriteAuxColsDef<uint8_t>);
#pragma unroll
        for (size_t chunk_idx = 0; chunk_idx < DIGEST_MEMORY_OPS; ++chunk_idx) {
            const size_t aux_idx = (section_idx - 1) * DIGEST_MEMORY_OPS + chunk_idx;
            RowSlice aux_row = row.slice_from(
                COL_INDEX(DeferralOutputCols, write_bytes_aux) + chunk_idx * write_aux_stride
            );
            COL_WRITE_ARRAY(aux_row, MemoryWriteAuxColsDef, prev_data, write_aux[aux_idx].prev_data);
            mem_helper.fill(
                aux_row,
                write_aux[aux_idx].prev_timestamp,
                header.from_timestamp + 2 + static_cast<uint32_t>(OUTPUT_TOTAL_MEMORY_OPS) +
                    static_cast<uint32_t>(aux_idx)
            );
        }
    }

    Fp prev_capacity[DIGEST_SIZE];
    if (is_first) {
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            prev_capacity[i] = Fp::zero();
        }
    } else {
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            prev_capacity[i] = per_row[row_idx - 1].poseidon2_res[i];
        }
    }

    poseidon2_buffer.record(
        row.slice_from(COL_INDEX(DeferralOutputCols, sponge_inputs)),
        RowSlice(prev_capacity, 1),
        is_last
    );

    COL_WRITE_ARRAY(row, DeferralOutputCols, poseidon2_res, row_data.poseidon2_res);
}

extern "C" int _deferral_output_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    const uint8_t *d_raw_records,
    const DeferralOutputPerCall *d_per_call,
    const DeferralOutputPerRow *d_per_row,
    size_t num_valid,
    uint32_t *d_count,
    size_t num_def_circuits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *d_bitwise,
    size_t address_bits,
    Fp *d_poseidon2_records,
    DeferralPoseidon2Count *d_poseidon2_counts,
    uint32_t *d_poseidon2_idx,
    size_t poseidon2_capacity,
    cudaStream_t stream
) {
    auto [grid, block] = kernel_launch_params(height, 256);
    assert(width == sizeof(DeferralOutputCols<uint8_t>));

    // poseidon2_capacity arrives from Rust in units of Fp elements; convert to record count.
    assert(poseidon2_capacity % 16 == 0 && "poseidon2_capacity must be a multiple of 16");
    size_t poseidon2_record_capacity = poseidon2_capacity / 16;

    deferral_output_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_raw_records,
        d_per_call,
        d_per_row,
        num_valid,
        d_count,
        num_def_circuits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        d_bitwise,
        address_bits,
        reinterpret_cast<FpArray<16> *>(d_poseidon2_records),
        d_poseidon2_counts,
        d_poseidon2_idx,
        poseidon2_record_capacity
    );
    return CHECK_KERNEL();
}
