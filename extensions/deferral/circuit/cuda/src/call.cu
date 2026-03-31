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

template <typename T> using MemoryWriteAuxCols4 = MemoryWriteAuxCols<T, MEMORY_OP_SIZE>;

__device__ __forceinline__ Fp bytes4_to_fp(const uint8_t *bytes) {
    const uint32_t value =
        static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    return Fp(value);
}

template <typename B, typename F> struct DeferralCallReads {
    B input_commit[COMMIT_NUM_BYTES];
    F old_input_acc[DIGEST_SIZE];
    F old_output_acc[DIGEST_SIZE];
};

template <typename B, typename F> struct DeferralCallWrites {
    B output_commit[COMMIT_NUM_BYTES];
    B output_len[F_NUM_BYTES];
    F new_input_acc[DIGEST_SIZE];
    F new_output_acc[DIGEST_SIZE];
};

// =========================== CORE ===============================

template <typename T> struct DeferralCallCoreRecord {
    T deferral_idx;
    DeferralCallReads<uint8_t, T> reads;
    DeferralCallWrites<uint8_t, T> writes;
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
    BitwiseOperationLookup &bitwise_buffer,
    DeferralPoseidon2Buffer &poseidon2_buffer,
    const size_t address_bits
) {
    COL_WRITE_VALUE(row, DeferralCallCoreCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, DeferralCallCoreCols, deferral_idx, record.deferral_idx);

    COL_WRITE_ARRAY(row, DeferralCallCoreCols, reads.input_commit, record.reads.input_commit);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, reads.old_input_acc, record.reads.old_input_acc);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, reads.old_output_acc, record.reads.old_output_acc);

    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.output_commit, record.writes.output_commit);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.output_len, record.writes.output_len);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.new_input_acc, record.writes.new_input_acc);
    COL_WRITE_ARRAY(row, DeferralCallCoreCols, writes.new_output_acc, record.writes.new_output_acc);

    count_buffer.add_count(record.deferral_idx.asUInt32());

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

#pragma unroll
    for (size_t i = 0; i < COMMIT_NUM_BYTES; i += 2) {
        bitwise_buffer.add_range(
            record.writes.output_commit[i], record.writes.output_commit[i + 1]
        );
    }
#pragma unroll
    for (size_t i = 0; i < F_NUM_BYTES; i += 2) {
        bitwise_buffer.add_range(record.writes.output_len[i], record.writes.output_len[i + 1]);
    }

    const uint32_t limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits;
    bitwise_buffer.add_range(
        static_cast<uint32_t>(record.writes.output_len[RV32_REGISTER_NUM_LIMBS - 1])
            << limb_shift_bits,
        0
    );

    constexpr size_t input_aux_offset = COL_INDEX(DeferralCallCoreCols, input_commit_lt_aux);
    constexpr size_t output_aux_offset = COL_INDEX(DeferralCallCoreCols, output_commit_lt_aux);
    constexpr size_t canonicity_aux_stride = sizeof(CanonicityAuxCols<uint8_t>);

    uint32_t input_commit_rcs[DIGEST_SIZE];
    uint32_t output_commit_rcs[DIGEST_SIZE];

#pragma unroll
    for (size_t i = 0; i < DIGEST_SIZE; ++i) {
        CanonicityAuxCols<Fp> aux;
        Fp x_le[F_NUM_BYTES];
#pragma unroll
        for (size_t j = 0; j < F_NUM_BYTES; ++j) {
            x_le[j] = Fp(record.reads.input_commit[i * F_NUM_BYTES + j]);
        }
        input_commit_rcs[i] = generate_subrow(x_le, aux);
        RowSlice aux_row = row.slice_from(input_aux_offset + i * canonicity_aux_stride);
        COL_WRITE_ARRAY(aux_row, CanonicityAuxCols, diff_marker, aux.diff_marker);
        COL_WRITE_VALUE(aux_row, CanonicityAuxCols, diff_val, aux.diff_val);
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_SIZE; ++i) {
        CanonicityAuxCols<Fp> aux;
        Fp x_le[F_NUM_BYTES];
#pragma unroll
        for (size_t j = 0; j < F_NUM_BYTES; ++j) {
            x_le[j] = Fp(record.writes.output_commit[i * F_NUM_BYTES + j]);
        }
        output_commit_rcs[i] = generate_subrow(x_le, aux);
        RowSlice aux_row = row.slice_from(output_aux_offset + i * canonicity_aux_stride);
        COL_WRITE_ARRAY(aux_row, CanonicityAuxCols, diff_marker, aux.diff_marker);
        COL_WRITE_VALUE(aux_row, CanonicityAuxCols, diff_val, aux.diff_val);
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_SIZE; i += 2) {
        bitwise_buffer.add_range(input_commit_rcs[i], input_commit_rcs[i + 1]);
        bitwise_buffer.add_range(output_commit_rcs[i], output_commit_rcs[i + 1]);
    }
}

// =========================== ADAPTER ============================

template <typename T> struct DeferralCallAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    T rd_ptr;
    T rs_ptr;

    uint8_t rd_val[RV32_REGISTER_NUM_LIMBS];
    uint8_t rs_val[RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxRecord rd_aux;
    MemoryReadAuxRecord rs_aux;

    MemoryReadAuxRecord input_commit_aux[COMMIT_MEMORY_OPS];
    MemoryReadAuxRecord old_input_acc_aux[DIGEST_MEMORY_OPS];
    MemoryReadAuxRecord old_output_acc_aux[DIGEST_MEMORY_OPS];

    MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE> output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];
    MemoryWriteAuxRecord<T, MEMORY_OP_SIZE> new_input_acc_aux[DIGEST_MEMORY_OPS];
    MemoryWriteAuxRecord<T, MEMORY_OP_SIZE> new_output_acc_aux[DIGEST_MEMORY_OPS];
};

template <typename T> struct DeferralCallAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs_ptr;

    T rd_val[RV32_REGISTER_NUM_LIMBS];
    T rs_val[RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> rd_aux;
    MemoryReadAuxCols<T> rs_aux;

    MemoryReadAuxCols<T> input_commit_aux[COMMIT_MEMORY_OPS];
    MemoryReadAuxCols<T> old_input_acc_aux[DIGEST_MEMORY_OPS];
    MemoryReadAuxCols<T> old_output_acc_aux[DIGEST_MEMORY_OPS];

    MemoryWriteAuxCols<T, MEMORY_OP_SIZE> output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];
    MemoryWriteAuxCols<T, MEMORY_OP_SIZE> new_input_acc_aux[DIGEST_MEMORY_OPS];
    MemoryWriteAuxCols<T, MEMORY_OP_SIZE> new_output_acc_aux[DIGEST_MEMORY_OPS];
};

__device__ __forceinline__ void deferral_call_adapter_tracegen(
    RowSlice row,
    const DeferralCallAdapterRecord<Fp> &record,
    BitwiseOperationLookup &bitwise_buffer,
    MemoryAuxColsFactory &mem_helper,
    const size_t address_bits
) {
    const uint32_t limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits;
    bitwise_buffer.add_range(
        static_cast<uint32_t>(record.rd_val[RV32_REGISTER_NUM_LIMBS - 1]) << limb_shift_bits,
        static_cast<uint32_t>(record.rs_val[RV32_REGISTER_NUM_LIMBS - 1]) << limb_shift_bits
    );

    COL_WRITE_VALUE(row, DeferralCallAdapterCols, from_state.pc, record.from_pc);
    COL_WRITE_VALUE(row, DeferralCallAdapterCols, from_state.timestamp, record.from_timestamp);
    COL_WRITE_VALUE(row, DeferralCallAdapterCols, rd_ptr, record.rd_ptr);
    COL_WRITE_VALUE(row, DeferralCallAdapterCols, rs_ptr, record.rs_ptr);
    COL_WRITE_ARRAY(row, DeferralCallAdapterCols, rd_val, record.rd_val);
    COL_WRITE_ARRAY(row, DeferralCallAdapterCols, rs_val, record.rs_val);

    uint32_t timestamp = record.from_timestamp;
    constexpr size_t read_aux_stride = sizeof(MemoryReadAuxCols<uint8_t>);
    constexpr size_t write_aux_stride = sizeof(MemoryWriteAuxCols4<uint8_t>);

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
    for (size_t i = 0; i < DIGEST_MEMORY_OPS; ++i) {
        mem_helper.fill(
            row.slice_from(
                COL_INDEX(DeferralCallAdapterCols, old_input_acc_aux) + i * read_aux_stride
            ),
            record.old_input_acc_aux[i].prev_timestamp,
            timestamp++
        );
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_MEMORY_OPS; ++i) {
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
            COL_INDEX(DeferralCallAdapterCols, output_commit_and_len_aux) + i * write_aux_stride
        );
        COL_WRITE_ARRAY(
            aux_row, MemoryWriteAuxCols4, prev_data, record.output_commit_and_len_aux[i].prev_data
        );
        mem_helper.fill(aux_row, record.output_commit_and_len_aux[i].prev_timestamp, timestamp++);
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_MEMORY_OPS; ++i) {
        RowSlice aux_row = row.slice_from(
            COL_INDEX(DeferralCallAdapterCols, new_input_acc_aux) + i * write_aux_stride
        );
        COL_WRITE_ARRAY(
            aux_row, MemoryWriteAuxCols4, prev_data, record.new_input_acc_aux[i].prev_data
        );
        mem_helper.fill(aux_row, record.new_input_acc_aux[i].prev_timestamp, timestamp++);
    }

#pragma unroll
    for (size_t i = 0; i < DIGEST_MEMORY_OPS; ++i) {
        RowSlice aux_row = row.slice_from(
            COL_INDEX(DeferralCallAdapterCols, new_output_acc_aux) + i * write_aux_stride
        );
        COL_WRITE_ARRAY(
            aux_row, MemoryWriteAuxCols4, prev_data, record.new_output_acc_aux[i].prev_data
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
    uint32_t *bitwise_ptr,
    const size_t bitwise_num_bits,
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
    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
    );
    BitwiseOperationLookup bitwise_buffer(bitwise_ptr, bitwise_num_bits);
    DeferralPoseidon2Buffer poseidon2_buffer(
        poseidon2_records,
        poseidon2_counts,
        poseidon2_idx,
        poseidon2_capacity
    );

    deferral_call_adapter_tracegen(row, record.adapter, bitwise_buffer, mem_helper, address_bits);
    deferral_call_core_tracegen(
        row.slice_from(COL_INDEX(DeferralCallCols, core)),
        record.core,
        count_buffer,
        bitwise_buffer,
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
    uint32_t *d_bitwise,
    uint32_t bitwise_num_bits,
    Fp *d_poseidon2_records,
    DeferralPoseidon2Count *d_poseidon2_counts,
    uint32_t *d_poseidon2_idx,
    size_t poseidon2_capacity,
    size_t address_bits
) {
    auto [grid, block] = kernel_launch_params(height);
    assert(width == sizeof(DeferralCallCols<uint8_t>));

    // poseidon2_capacity arrives from Rust in units of Fp elements; convert to record count.
    assert(poseidon2_capacity % 16 == 0 && "poseidon2_capacity must be a multiple of 16");
    size_t poseidon2_record_capacity = poseidon2_capacity / 16;

    deferral_call_tracegen<<<grid, block>>>(
        d_trace,
        height,
        reinterpret_cast<const DeferralCallRecord<Fp> *>(d_records),
        num_records,
        d_count,
        num_def_circuits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        d_bitwise,
        bitwise_num_bits,
        reinterpret_cast<FpArray<16> *>(d_poseidon2_records),
        d_poseidon2_counts,
        d_poseidon2_idx,
        poseidon2_record_capacity,
        address_bits
    );
    return CHECK_KERNEL();
}
