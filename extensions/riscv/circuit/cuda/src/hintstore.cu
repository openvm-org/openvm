#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "riscv/rvr_g2_trace.cuh"

using namespace riscv;
using namespace program;
using hintstore::MAX_HINT_BUFFER_DWORDS;
using hintstore::MAX_HINT_BUFFER_DWORDS_BITS;

static_assert(MAX_HINT_BUFFER_DWORDS_BITS <= U16_BITS, "rem_words must fit in one u16 cell");
constexpr uint32_t REM_WORDS_SHIFT = U16_BITS - MAX_HINT_BUFFER_DWORDS_BITS;

template <typename T> struct Rv64HintStoreCols {
    // common
    T is_single;
    T is_buffer;

    // Low u16 cell of the 8-byte RV64 register that holds `rem_words`; the upper register cells
    // are known to be zero and are hardcoded in the memory bus interaction.
    T rem_words;

    ExecutionState<T> from_state;
    T mem_ptr_ptr;
    // Low 32 bits of the 8-byte RV64 register that holds `mem_ptr`; the upper 4 bytes are
    // known to be zero and are hardcoded in the memory bus interaction.
    T mem_ptr_limbs[RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> mem_ptr_aux_cols;

    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    // One hint word packed as u16 cells.
    T data[BLOCK_FE_WIDTH];

    // only buffer
    T is_buffer_start;
    T num_words_ptr;
    MemoryReadAuxCols<T> num_words_aux_cols;
};

// This is the part of the record that we keep only once per instruction
struct Rv64HintStoreRecordHeader {
    uint32_t num_words;

    uint32_t from_pc;
    uint32_t timestamp;

    uint32_t mem_ptr_ptr;
    uint32_t mem_ptr;
    MemoryReadAuxRecord mem_ptr_aux_record;

    // will set `num_words_ptr` to `u32::MAX` in case of single hint
    uint32_t num_words_ptr;
    MemoryReadAuxRecord num_words_read;
};

// This is the part of the record that we keep `num_words` times per instruction
struct Rv64HintStoreVars {
    MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS> write_aux;
    uint8_t data[RV64_REGISTER_NUM_LIMBS];
};

struct Rv64HintStore {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64HintStore(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        Rv64HintStoreRecordHeader &record,
        Rv64HintStoreVars &write,
        uint32_t local_idx
    ) {
        bool is_single = record.num_words_ptr == UINT32_MAX;
        uint32_t timestamp = record.timestamp + local_idx * 3;
        uint32_t rem_words = record.num_words - local_idx;
        uint32_t mem_ptr = record.mem_ptr + local_idx * (uint32_t)RV64_REGISTER_NUM_LIMBS;
        uint32_t mem_ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(mem_ptr_limbs, mem_ptr);

        COL_WRITE_VALUE(row, Rv64HintStoreCols, is_single, is_single);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, is_buffer, !is_single);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, rem_words, rem_words);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, from_state.timestamp, timestamp);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, mem_ptr_ptr, record.mem_ptr_ptr);
        COL_WRITE_ARRAY(row, Rv64HintStoreCols, mem_ptr_limbs, mem_ptr_limbs);

        if (local_idx == 0) {
#ifdef CUDA_DEBUG
            // The overflow check for mem_ptr + num_words * 8 is not needed because
            // 8 * MAX_HINT_BUFFER_DWORDS < 2^pointer_max_bits guarantees no overflow
            assert(MAX_HINT_BUFFER_DWORDS_BITS + 3 < pointer_max_bits);
            assert(record.num_words <= MAX_HINT_BUFFER_DWORDS);
#endif

            // Range check for mem_ptr (using pointer_max_bits).
            uint32_t mem_ptr_shift = RV64_PTR_BITS - (uint32_t)pointer_max_bits;
            uint32_t mem_ptr_high_u16 = record.mem_ptr >> U16_BITS;
            range_checker.add_count(mem_ptr_high_u16 << mem_ptr_shift, U16_BITS);

            // Range check for num_words (using MAX_HINT_BUFFER_DWORDS_BITS).
            range_checker.add_count(record.num_words << REM_WORDS_SHIFT, U16_BITS);

            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64HintStoreCols, mem_ptr_aux_cols)),
                record.mem_ptr_aux_record.prev_timestamp,
                timestamp
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64HintStoreCols, mem_ptr_aux_cols)));
        }

        if (local_idx == 0 && !is_single) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64HintStoreCols, num_words_aux_cols)),
                record.num_words_read.prev_timestamp,
                timestamp + 1
            );
            COL_WRITE_VALUE(row, Rv64HintStoreCols, is_buffer_start, 1);
            COL_WRITE_VALUE(row, Rv64HintStoreCols, num_words_ptr, record.num_words_ptr);
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64HintStoreCols, num_words_aux_cols)));
            COL_WRITE_VALUE(row, Rv64HintStoreCols, is_buffer_start, 0);
            COL_WRITE_VALUE(row, Rv64HintStoreCols, num_words_ptr, 0);
        }

        Fp packed_prev[BLOCK_FE_WIDTH];
        pack_u8_block_bytes(packed_prev, write.write_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64HintStoreCols, write_aux.prev_data, packed_prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64HintStoreCols, write_aux)),
            write.write_aux.prev_timestamp,
            timestamp + 2
        );

        Fp packed_data[BLOCK_FE_WIDTH];
        pack_u8_block_bytes(packed_data, write.data);
        COL_WRITE_ARRAY(row, Rv64HintStoreCols, data, packed_data);
    }
};

struct OffsetInfo {
    uint32_t record_offset;
    uint32_t local_idx;
};

using HintReplayRow = riscv::G2HintReplayRow;

__global__ void hintstore_decode_offsets(
    const uint8_t *records,
    size_t records_len,
    size_t rows_used,
    OffsetInfo *record_offsets,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    size_t record_offset = 0;
    size_t row = 0;
    while (record_offset < records_len) {
        if (records_len - record_offset < sizeof(Rv64HintStoreRecordHeader)) {
            *error = 1;
            return;
        }
        auto header = reinterpret_cast<const Rv64HintStoreRecordHeader *>(
            records + record_offset
        );
        uint32_t num_words = header->num_words;
        if (num_words == 0 || num_words > MAX_HINT_BUFFER_DWORDS) {
            *error = 2;
            return;
        }
        size_t record_bytes = sizeof(Rv64HintStoreRecordHeader) +
                              (size_t)num_words * sizeof(Rv64HintStoreVars);
        if (record_bytes > records_len - record_offset) {
            *error = 3;
            return;
        }
        if ((size_t)num_words > rows_used - row) {
            *error = 4;
            return;
        }
        for (uint32_t local_idx = 0; local_idx < num_words; local_idx++) {
            record_offsets[row++] = OffsetInfo{
                .record_offset = (uint32_t)record_offset,
                .local_idx = local_idx,
            };
        }
        record_offset += record_bytes;
    }
    if (record_offset != records_len || row != rows_used) {
        *error = 5;
    }
}

extern "C" int _hintstore_decode_offsets(
    const uint8_t *__restrict__ d_records,
    size_t records_len,
    size_t rows_used,
    OffsetInfo *__restrict__ d_record_offsets,
    uint32_t *__restrict__ d_error,
    cudaStream_t stream
) {
    if (records_len == 0 || rows_used == 0) {
        return records_len == 0 && rows_used == 0 ? int(cudaSuccess)
                                                  : int(cudaErrorInvalidValue);
    }
    if (d_records == nullptr || d_record_offsets == nullptr || d_error == nullptr) {
        return int(cudaErrorInvalidValue);
    }
    hintstore_decode_offsets<<<1, 1, 0, stream>>>(
        d_records,
        records_len,
        rows_used,
        d_record_offsets,
        d_error
    );
    return CHECK_KERNEL();
}

__global__ void hintstore_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    uint32_t rows_used,
    OffsetInfo *record_offsets,
    uint32_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < rows_used) {
        auto record_offset = record_offsets[idx].record_offset;
        auto local_idx = record_offsets[idx].local_idx;
        auto record_header =
            *reinterpret_cast<Rv64HintStoreRecordHeader *>(records + record_offset);

        auto writes_start = records + record_offset + sizeof(Rv64HintStoreRecordHeader);

        auto data_write = reinterpret_cast<Rv64HintStoreVars *>(writes_start)[local_idx];

        auto filler = Rv64HintStore(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        filler.fill_trace_row(row, record_header, data_write, local_idx);
    } else {
        row.fill_zero(0, sizeof(Rv64HintStoreCols<uint8_t>));
    }
}

extern "C" int _hintstore_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    uint8_t *__restrict__ d_records,
    uint32_t rows_used,
    OffsetInfo *__restrict__ d_record_offsets,
    uint32_t pointer_max_bits,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64HintStoreCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    hintstore_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        rows_used,
        d_record_offsets,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__device__ __forceinline__ bool hintstore_fill_replay_row(
    RowSlice row,
    HintReplayRow replay,
    uint32_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    if (replay.reserved != 0 || replay.local_idx >= replay.num_words) {
        atomicCAS(error, 0u, 1u);
        return false;
    }
    Rv64HintStoreRecordHeader header{
        replay.num_words,
        replay.from_pc,
        replay.timestamp,
        replay.mem_ptr_ptr,
        replay.mem_ptr,
        {replay.mem_ptr_prev_timestamp},
        replay.num_words_ptr,
        {replay.num_words_prev_timestamp},
    };
    Rv64HintStoreVars write{};
    write.write_aux.prev_timestamp = replay.write_prev_timestamp;
    memcpy(write.write_aux.prev_data, &replay.write_prev_data, sizeof(replay.write_prev_data));
    memcpy(write.data, &replay.data, sizeof(replay.data));
    auto filler = Rv64HintStore(
        pointer_max_bits,
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
        timestamp_max_bits
    );
    filler.fill_trace_row(row, header, write, replay.local_idx);
    return true;
}

__global__ void hintstore_replay_tracegen(
    Fp *trace,
    size_t height,
    HintReplayRow const *rows,
    uint32_t rows_used,
    uint32_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < rows_used) {
        hintstore_fill_replay_row(
            row,
            rows[idx],
            pointer_max_bits,
            range_checker_ptr,
            range_checker_num_bins,
            timestamp_max_bits,
            error
        );
    } else {
        row.fill_zero(0, sizeof(Rv64HintStoreCols<uint8_t>));
    }
}

__global__ void hintstore_g2_tracegen(
    Fp *trace,
    size_t height,
    riscv::G2TraceSource source,
    riscv::RvrOperandEntry const *operand_table,
    uint32_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < source.row_count) {
        HintReplayRow replay{};
        if (!riscv::g2_trace_hint(source, operand_table, idx, replay) ||
            !hintstore_fill_replay_row(
                row,
                replay,
                pointer_max_bits,
                range_checker_ptr,
                range_checker_num_bins,
                timestamp_max_bits,
                reinterpret_cast<uint32_t *>(source.error)
            ))
            row.fill_zero(0, sizeof(Rv64HintStoreCols<uint8_t>));
    } else {
        row.fill_zero(0, sizeof(Rv64HintStoreCols<uint8_t>));
    }
}

extern "C" int _hintstore_replay_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    uint8_t const *__restrict__ d_rows,
    size_t rows_used,
    uint32_t pointer_max_bits,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *__restrict__ d_error,
    cudaStream_t stream
) {
    if (width != sizeof(Rv64HintStoreCols<uint8_t>) ||
        rows_used > UINT32_MAX || rows_used > height ||
        (height != 0 && d_trace == nullptr) ||
        (rows_used != 0 && d_rows == nullptr) || d_error == nullptr) {
        return int(cudaErrorInvalidValue);
    }
    if (height == 0) return int(cudaSuccess);
    auto [grid, block] = kernel_launch_params(height);
    hintstore_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        reinterpret_cast<HintReplayRow const *>(d_rows),
        uint32_t(rows_used),
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        d_error
    );
    return CHECK_KERNEL();
}

extern "C" int _hintstore_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS) {
    if (width != sizeof(Rv64HintStoreCols<uint8_t>) || source.kind != 30 ||
        source.row_count > height || (height != 0 && trace == nullptr) ||
        (source.row_count != 0 && operand_table == nullptr) || source.error == 0) {
        return int(cudaErrorInvalidValue);
    }
    if (height == 0) return int(cudaSuccess);
    auto [grid, block] = kernel_launch_params(height);
    hintstore_g2_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        source,
        operand_table,
        uint32_t(pointer_max_bits),
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _hintstore_tracegen_g2_preload(Fp *trace, cudaStream_t stream) {
    cudaFuncAttributes attributes{};
    cudaError_t status = cudaFuncGetAttributes(&attributes, hintstore_g2_tracegen);
    if (status != cudaSuccess) return int(status);
    riscv::G2TraceSource source{};
    hintstore_g2_tracegen<<<1, 1, 0, stream>>>(
        trace, 0, source, nullptr, 0, nullptr, 0, 0
    );
    return CHECK_KERNEL();
}

extern "C" int _hintstore_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS) {
    if (width != sizeof(Rv64HintStoreCols<uint8_t>) || records.size % sizeof(HintReplayRow) != 0 ||
        records.size / sizeof(HintReplayRow) > UINT32_MAX ||
        records.size / sizeof(HintReplayRow) > height ||
        (height != 0 && trace == nullptr) ||
        (records.size != 0 && records.ptr == 0) || range_checker == nullptr) {
        return int(cudaErrorInvalidValue);
    }
    if (height == 0) return int(cudaSuccess);
    auto [grid, block] = kernel_launch_params(height);
    uint32_t *error = nullptr;
    cudaError_t status = cudaMallocAsync(&error, sizeof(uint32_t), stream);
    if (status != cudaSuccess) return int(status);
    status = cudaMemsetAsync(error, 0, sizeof(uint32_t), stream);
    if (status != cudaSuccess) {
        cudaFreeAsync(error, stream);
        return int(status);
    }
    hintstore_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        records.as_typed<HintReplayRow>().ptr,
        uint32_t(records.size / sizeof(HintReplayRow)),
        uint32_t(pointer_max_bits),
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        error
    );
    status = cudaGetLastError();
    cudaError_t free_status = cudaFreeAsync(error, stream);
    return int(status != cudaSuccess ? status : free_status);
}
