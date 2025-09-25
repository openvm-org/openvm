#include "launcher.cuh"
#include "native/field_ext_operations.cuh"
#include "native/fri.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

using namespace riscv;
using namespace native;
using namespace program;

struct FriReducedOpeningRecordHeader {
    uint32_t length;
    bool is_init;
};

template <typename F> struct FriReducedOpeningCommonRecord {
    uint32_t timestamp;
    uint32_t a_ptr;
    uint32_t b_ptr;
    F alpha[EXT_DEG];
    uint32_t from_pc;
    F a_ptr_ptr;
    MemoryReadAuxRecord a_ptr_aux;
    F b_ptr_ptr;
    MemoryReadAuxRecord b_ptr_aux;
    F length_ptr;
    MemoryReadAuxRecord length_aux;
    F alpha_ptr;
    MemoryReadAuxRecord alpha_aux;
    F result_ptr;
    MemoryWriteAuxRecord<F, EXT_DEG> result_aux;
    F hint_id_ptr;
    F is_init_ptr;
    MemoryReadAuxRecord is_init_aux;
};

// Part of record for each workload row that calculates the partial `result`
template <typename F> struct FriReducedOpeningWorkloadRowRecord {
    F a;
    MemoryReadAuxRecord a_aux;
    F result[EXT_DEG];
    MemoryReadAuxRecord b_aux;
};

struct FriReducedOpening {
    MemoryAuxColsFactory mem_helper;

    __device__ FriReducedOpening(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_instruction1_row(
        RowSlice row,
        FriReducedOpeningCommonRecord<Fp> &common_rec,
        FriReducedOpeningRecordHeader &header,
        Fp *result
    ) {
        COL_WRITE_VALUE(row, Instruction1Cols, prefix.general.is_workload_row, Fp::zero());
        COL_WRITE_VALUE(row, Instruction1Cols, prefix.general.is_ins_row, Fp::one());
        COL_WRITE_VALUE(row, Instruction1Cols, prefix.general.timestamp, common_rec.timestamp);

        COL_WRITE_VALUE(row, Instruction1Cols, prefix.a_or_is_first, Fp::one());

        COL_WRITE_VALUE(row, Instruction1Cols, prefix.data.a_ptr, common_rec.a_ptr);
        COL_WRITE_VALUE(row, Instruction1Cols, prefix.data.write_a, !header.is_init);
        COL_WRITE_VALUE(row, Instruction1Cols, prefix.data.b_ptr, common_rec.b_ptr);
        COL_WRITE_VALUE(row, Instruction1Cols, prefix.data.idx, header.length);
        COL_WRITE_ARRAY(row, Instruction1Cols, prefix.data.result, result);
        COL_WRITE_ARRAY(row, Instruction1Cols, prefix.data.alpha, common_rec.alpha);

        COL_WRITE_VALUE(row, Instruction1Cols, pc, common_rec.from_pc);

        COL_WRITE_VALUE(row, Instruction1Cols, a_ptr_ptr, common_rec.a_ptr_ptr);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Instruction1Cols, a_ptr_aux)),
            common_rec.a_ptr_aux.prev_timestamp,
            common_rec.timestamp + 2
        );

        COL_WRITE_VALUE(row, Instruction1Cols, b_ptr_ptr, common_rec.b_ptr_ptr);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Instruction1Cols, b_ptr_aux)),
            common_rec.b_ptr_aux.prev_timestamp,
            common_rec.timestamp + 3
        );
        COL_WRITE_VALUE(row, Instruction1Cols, write_a_x_is_first, !header.is_init);
        row.fill_zero(INSN1_SIZE, OVERALL_SIZE - INSN1_SIZE);
    }

    __device__ void fill_instruction2_row(
        RowSlice row,
        FriReducedOpeningCommonRecord<Fp> &common_rec,
        FriReducedOpeningRecordHeader &header
    ) {
        COL_WRITE_VALUE(row, Instruction2Cols, general.is_workload_row, Fp::zero());
        COL_WRITE_VALUE(row, Instruction2Cols, general.is_ins_row, Fp::one());
        COL_WRITE_VALUE(row, Instruction2Cols, general.timestamp, common_rec.timestamp);

        COL_WRITE_VALUE(row, Instruction2Cols, is_first, Fp::zero());

        COL_WRITE_VALUE(row, Instruction2Cols, length_ptr, common_rec.length_ptr);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Instruction2Cols, length_aux)),
            common_rec.length_aux.prev_timestamp,
            common_rec.timestamp + 1
        );
        COL_WRITE_VALUE(row, Instruction2Cols, alpha_ptr, common_rec.alpha_ptr);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Instruction2Cols, alpha_aux)),
            common_rec.alpha_aux.prev_timestamp,
            common_rec.timestamp
        );
        COL_WRITE_VALUE(row, Instruction2Cols, result_ptr, common_rec.result_ptr);
        COL_WRITE_ARRAY(
            row, Instruction2Cols, result_aux.prev_data, common_rec.result_aux.prev_data
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Instruction2Cols, result_aux)),
            common_rec.result_aux.prev_timestamp,
            common_rec.timestamp + 5 + 2 * header.length
        );
        COL_WRITE_VALUE(row, Instruction2Cols, hint_id_ptr, common_rec.hint_id_ptr);
        COL_WRITE_VALUE(row, Instruction2Cols, is_init_ptr, common_rec.is_init_ptr);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Instruction2Cols, is_init_aux)),
            common_rec.is_init_aux.prev_timestamp,
            common_rec.timestamp + 4
        );
        COL_WRITE_VALUE(row, Instruction2Cols, write_a_x_is_first, Fp::zero());
        row.fill_zero(INSN2_SIZE, OVERALL_SIZE - INSN2_SIZE);
    }

    // If local_idx == 0, then `prev_result` is not used and it can be nullptr
    // If is_init is true, then `prev_data` is not used and it can be nullptr
    __device__ void fill_workload_row(
        RowSlice row,
        FriReducedOpeningCommonRecord<Fp> &common_rec,
        FriReducedOpeningWorkloadRowRecord<Fp> &wl_rec,
        FriReducedOpeningRecordHeader &header,
        Fp *prev_data,
        Fp *prev_result,
        uint32_t local_idx
    ) {
        uint32_t local_idx_rev = header.length - local_idx;
        uint32_t timestamp = common_rec.timestamp + local_idx_rev * 2;
        COL_WRITE_VALUE(row, WorkloadCols, prefix.general.is_workload_row, Fp::one());
        COL_WRITE_VALUE(row, WorkloadCols, prefix.general.is_ins_row, Fp::zero());
        COL_WRITE_VALUE(row, WorkloadCols, prefix.general.timestamp, timestamp);

        COL_WRITE_VALUE(row, WorkloadCols, prefix.a_or_is_first, wl_rec.a);

        COL_WRITE_VALUE(row, WorkloadCols, prefix.data.a_ptr, common_rec.a_ptr + local_idx_rev);
        COL_WRITE_VALUE(row, WorkloadCols, prefix.data.write_a, !header.is_init);
        COL_WRITE_VALUE(
            row, WorkloadCols, prefix.data.b_ptr, common_rec.b_ptr + local_idx_rev * EXT_DEG
        );
        COL_WRITE_VALUE(row, WorkloadCols, prefix.data.idx, local_idx);
        if (local_idx == 0) {
            COL_FILL_ZERO(row, WorkloadCols, prefix.data.result);
        } else {
            COL_WRITE_ARRAY(row, WorkloadCols, prefix.data.result, prev_result);
        }
        COL_WRITE_ARRAY(row, WorkloadCols, prefix.data.alpha, common_rec.alpha);

        if (header.is_init) {
            COL_WRITE_VALUE(row, WorkloadCols, a_aux.prev_data, Fp::zero());
        } else {
            COL_WRITE_VALUE(row, WorkloadCols, a_aux.prev_data, prev_data[local_idx]);
        }

        mem_helper.fill(
            row.slice_from(COL_INDEX(WorkloadCols, a_aux)),
            wl_rec.a_aux.prev_timestamp,
            timestamp + 3
        );

        // Reorder the formula: result = prev_result * alpha + (b - a)
        // to get: b = result + a - prev_result * alpha
        if (local_idx > 0) {
            FieldExtElement<Fp> x{common_rec.alpha}, y{prev_result};
            auto b = FieldExtOperations::subtract(
                FieldExtOperations::add(wl_rec.result, wl_rec.a), FieldExtOperations::multiply(x, y)
            );
            COL_WRITE_ARRAY(row, WorkloadCols, b, b.el);
        } else {
            auto b = FieldExtOperations::add(wl_rec.result, wl_rec.a);
            COL_WRITE_ARRAY(row, WorkloadCols, b, b.el);
        }

        mem_helper.fill(
            row.slice_from(COL_INDEX(WorkloadCols, b_aux)),
            wl_rec.b_aux.prev_timestamp,
            timestamp + 4
        );
        row.fill_zero(WORKLOAD_SIZE, OVERALL_SIZE - WORKLOAD_SIZE);
    }
};

struct RowInfo {
    uint32_t record_offset;
    uint32_t local_idx;
};

__global__ void fri_reduced_opening_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t rows_used,
    RowInfo *rows_info,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows_used) {
        // The record consists of (in order):
        // - a header [FriReducedOpeningRecordHeader]
        // - `header.length` times [FriReducedOpeningWorkloadRowRecord]s
        // - iff header.is_init is false: `header.length` times [Fp]s (this represents the previous data)
        // - a common record [FriReducedOpeningCommonRecord]
        auto record_offset = rows_info[idx].record_offset;
        auto local_idx = rows_info[idx].local_idx;
        auto header = *reinterpret_cast<FriReducedOpeningRecordHeader *>(records + record_offset);

        uint32_t wl_size = header.length * sizeof(FriReducedOpeningWorkloadRowRecord<Fp>);
        uint32_t prev_data_size = header.is_init ? 0 : header.length * sizeof(Fp);

        auto wl_start = records + record_offset + sizeof(FriReducedOpeningRecordHeader);
        auto prev_data_start = wl_start + wl_size;
        auto common_rec_start = prev_data_start + prev_data_size;

        auto common_rec = *reinterpret_cast<FriReducedOpeningCommonRecord<Fp> *>(common_rec_start);

        auto step = FriReducedOpening(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        RowSlice row(trace + idx, height);
        if (local_idx == header.length) {
            auto wl_rec = reinterpret_cast<FriReducedOpeningWorkloadRowRecord<Fp> *>(wl_start
            )[header.length - 1];
            step.fill_instruction1_row(row, common_rec, header, wl_rec.result);
        } else if (local_idx == header.length + 1) {
            step.fill_instruction2_row(row, common_rec, header);
        } else {
            auto wl_rec =
                reinterpret_cast<FriReducedOpeningWorkloadRowRecord<Fp> *>(wl_start)[local_idx];
            Fp *prev_result = nullptr;
            if (local_idx > 0) {
                auto prev_wl_rec =
                    reinterpret_cast<FriReducedOpeningWorkloadRowRecord<Fp> *>(wl_start
                    )[local_idx - 1];
                prev_result = prev_wl_rec.result;
            }
            Fp *prev_data = header.is_init ? nullptr : reinterpret_cast<Fp *>(prev_data_start);
            step.fill_workload_row(
                row, common_rec, wl_rec, header, prev_data, prev_result, local_idx
            );
        }
    } else {
        // Fill with 0s
        RowSlice row(trace + idx, height);
        row.fill_zero(0, OVERALL_SIZE);
    }
}

extern "C" int _fri_reduced_opening_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t rows_used,
    RowInfo *d_rows_info,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    auto [grid, block] = kernel_launch_params(height, 512);
    fri_reduced_opening_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        rows_used,
        d_rows_info,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
