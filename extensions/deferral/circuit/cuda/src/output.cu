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

template <typename T> using MemoryWriteAuxColsDef = MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>;

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

    // Heap pointers + auxiliary read columns.
    // Low 32 bits of heap pointers, packed as u16 cells.
    T rd_val[RV64_PTR_U16S];
    T rs_val[RV64_PTR_U16S];
    MemoryReadAuxCols<T> rd_aux;
    MemoryReadAuxCols<T> rs_aux;

    // First row reads [output_commit || output_len_le] from heap as u16 cells.
    // `output_commit` is the onion hash of the output bytes.
    T output_commit[COMMIT_NUM_U16S];
    T output_len[F_NUM_U16S];
    MemoryReadAuxCols<T> output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];

    // Auxiliary columns to ensure canonicity of output_commit cells.
    CanonicityAuxCols<T> output_commit_lt_aux[DIGEST_SIZE];

    // First row sponge input is [deferral_idx, output_len, 0, ...].
    // Later rows sponge and write the next SPONGE_BYTES_PER_ROW output bytes.
    T sponge_inputs[DIGEST_SIZE];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_bytes_aux[SPONGE_ROW_MEMORY_OPS];

    // Running Poseidon2 capacity on non-last rows; final compression on the last row.
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

    const uint32_t output_len = (header.num_rows - 1) * SPONGE_BYTES_PER_ROW;
    const bool is_first = section_idx == 0;
    const bool is_last = section_idx + 1 == header.num_rows;

    Histogram count_buffer(count_ptr, num_def_circuits);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);
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

    Fp rd_val_u16s[RV64_PTR_U16S];
    Fp rs_val_u16s[RV64_PTR_U16S];
    u32_bytes_to_le_u16_cells(rd_val_u16s, header.rd_val);
    u32_bytes_to_le_u16_cells(rs_val_u16s, header.rs_val);
    COL_WRITE_ARRAY(row, DeferralOutputCols, rd_val, rd_val_u16s);
    COL_WRITE_ARRAY(row, DeferralOutputCols, rs_val, rs_val_u16s);

    uint16_t output_commit_u16s[COMMIT_NUM_U16S];
    pack_u8_pairs_le(output_commit_u16s, call_data.output_commit);
    COL_WRITE_ARRAY(row, DeferralOutputCols, output_commit, output_commit_u16s);
    uint16_t output_len_u16s[F_NUM_U16S];
    u32_to_le_u16_cells(output_len_u16s, output_len);
    COL_WRITE_ARRAY(row, DeferralOutputCols, output_len, output_len_u16s);

    if (is_first) {
        count_buffer.add_count(header.deferral_idx);

        range_checker.add_count(
            scale_rv64_ptr_from_u32_bytes(header.rd_val, address_bits), U16_BITS
        );
        range_checker.add_count(
            scale_rv64_ptr_from_u32_bytes(header.rs_val, address_bits), U16_BITS
        );
        // Mirror the AIR's output_len pointer-width check.
        range_checker.add_count(
            scale_output_len(output_len_u16s, address_bits),
            U16_BITS
        );

        // Per-cell 16-bit range checks on `output_commit` and `output_len`.
#pragma unroll
        for (size_t i = 0; i < COMMIT_NUM_U16S; ++i) {
            range_checker.add_count(static_cast<uint32_t>(output_commit_u16s[i]), U16_BITS);
        }
#pragma unroll
        for (size_t i = 0; i < F_NUM_U16S; ++i) {
            range_checker.add_count(static_cast<uint32_t>(output_len_u16s[i]), U16_BITS);
        }

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

#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            CanonicityAuxCols<Fp> aux;
            Fp x_le[F_NUM_U16S];
#pragma unroll
            for (size_t j = 0; j < F_NUM_U16S; ++j) {
                x_le[j] = Fp(static_cast<uint32_t>(output_commit_u16s[i * F_NUM_U16S + j]));
            }
            uint32_t rc = generate_subrow(x_le, aux);
            range_checker.add_count(rc, U16_BITS);
            write_canonicity_aux(row, COL_INDEX(DeferralOutputCols, output_commit_lt_aux), i, aux);
        }

        Fp sponge_cells[DIGEST_SIZE];
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            sponge_cells[i] = Fp::zero();
        }
        sponge_cells[0] = Fp(header.deferral_idx);
        sponge_cells[1] = Fp(output_len);
        COL_WRITE_ARRAY(row, DeferralOutputCols, sponge_inputs, sponge_cells);

        COL_FILL_ZERO(row, DeferralOutputCols, write_bytes_aux);
    } else {
        const uint8_t *header_end = record_start + sizeof(DeferralOutputRecordHeader);
        const uint8_t *write_bytes_start =
            header_end + (section_idx - 1) * SPONGE_BYTES_PER_ROW;
        const size_t write_aux_offset = align_up(
            sizeof(DeferralOutputRecordHeader) + output_len,
            alignof(MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES>)
        );
        const auto *write_aux = reinterpret_cast<const MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES> *>(
            record_start + write_aux_offset
        );

        COL_FILL_ZERO(row, DeferralOutputCols, rd_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, rs_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, output_commit_and_len_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, output_commit_lt_aux);

        // Pack each byte pair into one u16 sponge cell and range-check it.
        Fp sponge_cells[DIGEST_SIZE];
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            const size_t offset = U16_CELL_SIZE * i;
            const uint32_t cell = static_cast<uint32_t>(
                u16_from_bytes_le(write_bytes_start[offset], write_bytes_start[offset + 1])
            );
            sponge_cells[i] = Fp(cell);
            range_checker.add_count(cell, U16_BITS);
        }
        COL_WRITE_ARRAY(row, DeferralOutputCols, sponge_inputs, sponge_cells);

        constexpr size_t write_aux_stride = sizeof(MemoryWriteAuxColsDef<uint8_t>);
#pragma unroll
        for (size_t chunk_idx = 0; chunk_idx < SPONGE_ROW_MEMORY_OPS; ++chunk_idx) {
            const size_t aux_idx = (section_idx - 1) * SPONGE_ROW_MEMORY_OPS + chunk_idx;
            RowSlice aux_row = row.slice_from(
                COL_INDEX(DeferralOutputCols, write_bytes_aux) + chunk_idx * write_aux_stride
            );
            Fp packed_prev[BLOCK_FE_WIDTH];
            pack_u8_block_bytes(packed_prev, write_aux[aux_idx].prev_data);
            COL_WRITE_ARRAY(aux_row, MemoryWriteAuxColsDef, prev_data, packed_prev);
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
        address_bits,
        reinterpret_cast<FpArray<16> *>(d_poseidon2_records),
        d_poseidon2_counts,
        d_poseidon2_idx,
        poseidon2_record_capacity
    );
    return CHECK_KERNEL();
}
