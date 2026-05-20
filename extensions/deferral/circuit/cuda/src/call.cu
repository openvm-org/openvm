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

// Byte-AS write aux (RV64_MEMORY_AS): `MEMORY_BLOCK_BYTES` bytes packed into
// `BLOCK_FE_WIDTH` field cells per bus op.
template <typename T> using MemoryWriteAuxColsByte = MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>;
// F-AS write aux (DEFERRAL_AS): `BLOCK_FE_WIDTH` cells per bus op.
template <typename T> using MemoryWriteAuxColsF = MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>;

__device__ __forceinline__ Fp bytes4_to_fp(const uint8_t *bytes) {
    uint32_t value = 0;
#pragma unroll
    for (size_t i = 0; i < F_NUM_U16S; ++i) {
        const size_t offset = U16_CELL_SIZE * i;
        value |= static_cast<uint32_t>(u16_from_bytes_le(bytes[offset], bytes[offset + 1]))
                 << (U16_BITS * i);
    }
    return Fp(value);
}

// Byte-shaped record siblings; tracegen packs byte pairs into u16 columns.
template <typename F> struct DeferralCallReadsBytes {
    uint8_t input_commit[COMMIT_NUM_BYTES];
    F old_input_acc[DIGEST_SIZE];
    F old_output_acc[DIGEST_SIZE];
};

template <typename F> struct DeferralCallWritesBytes {
    uint8_t output_commit[COMMIT_NUM_BYTES];
    // Low 32 bits of the output length; padded to the 8-byte OutputKey field when written.
    uint8_t output_len[F_NUM_BYTES];
    F new_input_acc[DIGEST_SIZE];
    F new_output_acc[DIGEST_SIZE];
};

// u16-cell column shape.
template <typename B, typename F> struct DeferralCallReads {
    B input_commit[COMMIT_NUM_U16S];
    F old_input_acc[DIGEST_SIZE];
    F old_output_acc[DIGEST_SIZE];
};

template <typename B, typename F> struct DeferralCallWrites {
    B output_commit[COMMIT_NUM_U16S];
    B output_len[F_NUM_U16S];
    F new_input_acc[DIGEST_SIZE];
    F new_output_acc[DIGEST_SIZE];
};

// =========================== CORE ===============================

template <typename T> struct DeferralCallCoreRecord {
    T deferral_idx;
    DeferralCallReadsBytes<T> reads;
    DeferralCallWritesBytes<T> writes;
};

template <typename T> struct DeferralCallCoreCols {
    T is_valid;
    T deferral_idx;
    DeferralCallReads<T, T> reads;
    DeferralCallWrites<T, T> writes;

    CanonicityAuxCols<T> input_commit_lt_aux[DIGEST_SIZE];
    CanonicityAuxCols<T> output_commit_lt_aux[DIGEST_SIZE];
};

__device__ __forceinline__ void deferral_call_core_tracegen(
    RowSlice row,
    const DeferralCallCoreRecord<Fp> &record,
    Histogram &count_buffer,
    VariableRangeChecker &range_checker,
    DeferralPoseidon2Buffer &poseidon2_buffer,
    const size_t address_bits
) {
    COL_WRITE_VALUE(row, DeferralCallCoreCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, DeferralCallCoreCols, deferral_idx, record.deferral_idx);

    uint16_t input_commit_u16s[COMMIT_NUM_U16S];
    uint16_t output_commit_u16s[COMMIT_NUM_U16S];
    uint16_t output_len_u16s[F_NUM_U16S];
    pack_u8_pairs_le(input_commit_u16s, record.reads.input_commit);
    pack_u8_pairs_le(output_commit_u16s, record.writes.output_commit);
    pack_u8_pairs_le(output_len_u16s, record.writes.output_len);

    COL_WRITE_ARRAY(row, DeferralCallCoreCols, reads.input_commit, input_commit_u16s);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, reads.old_input_acc, record.reads.old_input_acc);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, reads.old_output_acc, record.reads.old_output_acc);

    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.output_commit, output_commit_u16s);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.output_len, output_len_u16s);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.new_input_acc, record.writes.new_input_acc);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.new_output_acc, record.writes.new_output_acc);

    count_buffer.add_count(record.deferral_idx.asUInt32());

    // Compute F-level commits from the raw byte record.
    Fp input_f_commit[DIGEST_SIZE];
    Fp output_f_commit[DIGEST_SIZE];
#pragma unroll
    for (size_t i = 0; i < DIGEST_SIZE; ++i) {
        input_f_commit[i] = bytes4_to_fp(record.reads.input_commit + i * F_NUM_BYTES);
        output_f_commit[i] = bytes4_to_fp(record.writes.output_commit + i * F_NUM_BYTES);
    }
    poseidon2_buffer.record(
        row.slice_from(COL_INDEX(DeferralCallCoreCols, reads.old_input_acc)),
        RowSlice(input_f_commit, 1),
        true
    );
    poseidon2_buffer.record(
        row.slice_from(COL_INDEX(DeferralCallCoreCols, reads.old_output_acc)),
        RowSlice(output_f_commit, 1),
        true
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

    range_checker.add_count(
        scale_output_len(output_len_u16s, address_bits),
        U16_BITS
    );

    constexpr size_t input_aux_offset = COL_INDEX(DeferralCallCoreCols, input_commit_lt_aux);
    constexpr size_t output_aux_offset = COL_INDEX(DeferralCallCoreCols, output_commit_lt_aux);
    constexpr size_t canonicity_aux_stride = sizeof(CanonicityAuxCols<uint8_t>);

#pragma unroll
    for (size_t i = 0; i < DIGEST_SIZE; ++i) {
        CanonicityAuxCols<Fp> aux;
        Fp x_le[F_NUM_U16S];
#pragma unroll
        for (size_t j = 0; j < F_NUM_U16S; ++j) {
            x_le[j] = Fp(static_cast<uint32_t>(input_commit_u16s[i * F_NUM_U16S + j]));
        }
        uint32_t rc = generate_subrow(x_le, aux);
        range_checker.add_count(rc, U16_BITS);
        RowSlice aux_row = row.slice_from(input_aux_offset + i * canonicity_aux_stride);
        COL_WRITE_ARRAY(aux_row, CanonicityAuxCols, diff_marker, aux.diff_marker);
        COL_WRITE_VALUE(aux_row, CanonicityAuxCols, diff_val, aux.diff_val);
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
        RowSlice aux_row = row.slice_from(output_aux_offset + i * canonicity_aux_stride);
        COL_WRITE_ARRAY(aux_row, CanonicityAuxCols, diff_marker, aux.diff_marker);
        COL_WRITE_VALUE(aux_row, CanonicityAuxCols, diff_val, aux.diff_val);
    }
}

// =========================== ADAPTER ============================

template <typename T> struct DeferralCallAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    T rd_ptr;
    T rs_ptr;

    uint8_t rd_val[RV64_WORD_NUM_LIMBS];
    uint8_t rs_val[RV64_WORD_NUM_LIMBS];
    MemoryReadAuxRecord rd_aux;
    MemoryReadAuxRecord rs_aux;

    MemoryReadAuxRecord input_commit_aux[COMMIT_MEMORY_OPS];
    MemoryReadAuxRecord old_input_acc_aux[DIGEST_F_MEMORY_OPS];
    MemoryReadAuxRecord old_output_acc_aux[DIGEST_F_MEMORY_OPS];

    MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES>
        output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];
    MemoryWriteAuxRecord<T, BLOCK_FE_WIDTH> new_input_acc_aux[DIGEST_F_MEMORY_OPS];
    MemoryWriteAuxRecord<T, BLOCK_FE_WIDTH> new_output_acc_aux[DIGEST_F_MEMORY_OPS];
};

template <typename T> struct DeferralCallAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs_ptr;

    // Low 32 bits of heap pointers, packed as u16 cells.
    T rd_val[RV64_PTR_U16S];
    T rs_val[RV64_PTR_U16S];
    MemoryReadAuxCols<T> rd_aux;
    MemoryReadAuxCols<T> rs_aux;

    MemoryReadAuxCols<T> input_commit_aux[COMMIT_MEMORY_OPS];
    MemoryReadAuxCols<T> old_input_acc_aux[DIGEST_F_MEMORY_OPS];
    MemoryReadAuxCols<T> old_output_acc_aux[DIGEST_F_MEMORY_OPS];

    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> new_input_acc_aux[DIGEST_F_MEMORY_OPS];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> new_output_acc_aux[DIGEST_F_MEMORY_OPS];
};

__device__ __forceinline__ void deferral_call_adapter_tracegen(
    RowSlice row,
    const DeferralCallAdapterRecord<Fp> &record,
    VariableRangeChecker &range_checker,
    MemoryAuxColsFactory &mem_helper,
    const size_t address_bits
) {
    range_checker.add_count(
        scale_rv64_ptr_from_u32_bytes(record.rd_val, address_bits), U16_BITS
    );
    range_checker.add_count(
        scale_rv64_ptr_from_u32_bytes(record.rs_val, address_bits), U16_BITS
    );

    COL_WRITE_VALUE(row, DeferralCallAdapterCols, from_state.pc, record.from_pc);
    COL_WRITE_VALUE(row, DeferralCallAdapterCols, from_state.timestamp, record.from_timestamp);
    COL_WRITE_VALUE(row, DeferralCallAdapterCols, rd_ptr, record.rd_ptr);
    COL_WRITE_VALUE(row, DeferralCallAdapterCols, rs_ptr, record.rs_ptr);

    Fp rd_val_u16s[RV64_PTR_U16S];
    Fp rs_val_u16s[RV64_PTR_U16S];
    u32_bytes_to_le_u16_cells(rd_val_u16s, record.rd_val);
    u32_bytes_to_le_u16_cells(rs_val_u16s, record.rs_val);
    COL_WRITE_ARRAY(row, DeferralCallAdapterCols, rd_val, rd_val_u16s);
    COL_WRITE_ARRAY(row, DeferralCallAdapterCols, rs_val, rs_val_u16s);

    uint32_t timestamp = record.from_timestamp;
    constexpr size_t read_aux_stride = sizeof(MemoryReadAuxCols<uint8_t>);
    constexpr size_t write_byte_aux_stride = sizeof(MemoryWriteAuxColsByte<uint8_t>);
    constexpr size_t write_f_aux_stride = sizeof(MemoryWriteAuxColsF<uint8_t>);

    mem_helper.fill(
        row.slice_from(COL_INDEX(DeferralCallAdapterCols, rd_aux)),
        record.rd_aux.prev_timestamp,
        timestamp++
    );
    mem_helper.fill(
        row.slice_from(COL_INDEX(DeferralCallAdapterCols, rs_aux)),
        record.rs_aux.prev_timestamp,
        timestamp++
    );

#pragma unroll
    for (size_t i = 0; i < COMMIT_MEMORY_OPS; ++i) {
        mem_helper.fill(
            row.slice_from(
                COL_INDEX(DeferralCallAdapterCols, input_commit_aux) + i * read_aux_stride
            ),
            record.input_commit_aux[i].prev_timestamp,
            timestamp++
        );
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_F_MEMORY_OPS; ++i) {
        mem_helper.fill(
            row.slice_from(
                COL_INDEX(DeferralCallAdapterCols, old_input_acc_aux) + i * read_aux_stride
            ),
            record.old_input_acc_aux[i].prev_timestamp,
            timestamp++
        );
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_F_MEMORY_OPS; ++i) {
        mem_helper.fill(
            row.slice_from(
                COL_INDEX(DeferralCallAdapterCols, old_output_acc_aux) + i * read_aux_stride
            ),
            record.old_output_acc_aux[i].prev_timestamp,
            timestamp++
        );
    }

#pragma unroll
    for (size_t i = 0; i < OUTPUT_TOTAL_MEMORY_OPS; ++i) {
        RowSlice aux_row = row.slice_from(
            COL_INDEX(DeferralCallAdapterCols, output_commit_and_len_aux) +
            i * write_byte_aux_stride
        );
        Fp packed_prev[BLOCK_FE_WIDTH];
        pack_u8_block_bytes(packed_prev, record.output_commit_and_len_aux[i].prev_data);
        COL_WRITE_ARRAY(aux_row, MemoryWriteAuxColsByte, prev_data, packed_prev);
        mem_helper.fill(aux_row, record.output_commit_and_len_aux[i].prev_timestamp, timestamp++);
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_F_MEMORY_OPS; ++i) {
        RowSlice aux_row = row.slice_from(
            COL_INDEX(DeferralCallAdapterCols, new_input_acc_aux) + i * write_f_aux_stride
        );
        COL_WRITE_ARRAY(
            aux_row, MemoryWriteAuxColsF, prev_data, record.new_input_acc_aux[i].prev_data
        );
        mem_helper.fill(aux_row, record.new_input_acc_aux[i].prev_timestamp, timestamp++);
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_F_MEMORY_OPS; ++i) {
        RowSlice aux_row = row.slice_from(
            COL_INDEX(DeferralCallAdapterCols, new_output_acc_aux) + i * write_f_aux_stride
        );
        COL_WRITE_ARRAY(
            aux_row, MemoryWriteAuxColsF, prev_data, record.new_output_acc_aux[i].prev_data
        );
        mem_helper.fill(aux_row, record.new_output_acc_aux[i].prev_timestamp, timestamp++);
    }
}

// =========================== COMBINED ===========================

template <typename T> struct DeferralCallRecord {
    DeferralCallAdapterRecord<T> adapter;
    DeferralCallCoreRecord<T> core;
};

template <typename T> struct DeferralCallCols {
    DeferralCallAdapterCols<T> adapter;
    DeferralCallCoreCols<T> core;
};

__global__ void deferral_call_tracegen(
    Fp *trace,
    const size_t height,
    const DeferralCallRecord<Fp> *records,
    const size_t num_records,
    uint32_t *count_ptr,
    const size_t num_def_circuits,
    uint32_t *range_checker_ptr,
    const uint32_t range_checker_num_bins,
    const uint32_t timestamp_max_bits,
    FpArray<16> *poseidon2_records,
    DeferralPoseidon2Count *poseidon2_counts,
    uint32_t *poseidon2_idx,
    const size_t poseidon2_capacity,
    const size_t address_bits
) {
    const uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_records) {
        row.fill_zero(0, sizeof(DeferralCallCols<uint8_t>));
        return;
    }

    DeferralCallRecord<Fp> record = records[row_idx];
    Histogram count_buffer(count_ptr, num_def_circuits);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);
    DeferralPoseidon2Buffer poseidon2_buffer(
        poseidon2_records, poseidon2_counts, poseidon2_idx, poseidon2_capacity
    );

    deferral_call_adapter_tracegen(row, record.adapter, range_checker, mem_helper, address_bits);
    deferral_call_core_tracegen(
        row.slice_from(COL_INDEX(DeferralCallCols, core)),
        record.core,
        count_buffer,
        range_checker,
        poseidon2_buffer,
        address_bits
    );
}

// =========================== LAUNCHER ===========================

extern "C" int _deferral_call_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    const uint8_t *d_records,
    size_t num_records,
    uint32_t *d_count,
    size_t num_def_circuits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    Fp *d_poseidon2_records,
    DeferralPoseidon2Count *d_poseidon2_counts,
    uint32_t *d_poseidon2_idx,
    size_t poseidon2_capacity,
    size_t address_bits,
    cudaStream_t stream
) {
    auto [grid, block] = kernel_launch_params(height, 256);
    assert(width == sizeof(DeferralCallCols<uint8_t>));

    // poseidon2_capacity arrives from Rust in units of Fp elements; convert to record count.
    assert(poseidon2_capacity % 16 == 0 && "poseidon2_capacity must be a multiple of 16");
    size_t poseidon2_record_capacity = poseidon2_capacity / 16;

    deferral_call_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        reinterpret_cast<const DeferralCallRecord<Fp> *>(d_records),
        num_records,
        d_count,
        num_def_circuits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        reinterpret_cast<FpArray<16> *>(d_poseidon2_records),
        d_poseidon2_counts,
        d_poseidon2_idx,
        poseidon2_record_capacity,
        address_bits
    );
    return CHECK_KERNEL();
}
