#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;
using namespace program;
using hintstore::MAX_HINT_BUFFER_DWORDS;
using hintstore::MAX_HINT_BUFFER_DWORDS_BITS;

// The rem_words range check below only works for MAX_HINT_BUFFER_DWORDS_BITS <= 16
static_assert(
    MAX_HINT_BUFFER_DWORDS_BITS <= 16,
    "MAX_HINT_BUFFER_DWORDS_BITS must be <= 16 for the rem_words range check"
);

// Number of u16 cells used to represent the low 32 bits of `mem_ptr`. Matches the Rust
// `MEM_PTR_NUM_LIMBS` constant.
constexpr size_t MEM_PTR_NUM_LIMBS = RV64_WORD_NUM_LIMBS / 2;

template <typename T> struct Rv64HintStoreCols {
    // common
    T is_single;
    T is_buffer;

    // Single u16 cell holding `rem_words`. `rem_words <= MAX_HINT_BUFFER_DWORDS < 2^16`
    // fits in a single u16 cell; the upper 6 bytes of the 8-byte RV64 register are zero
    // and hardcoded in the memory bus interaction.
    T rem_words;

    ExecutionState<T> from_state;
    T mem_ptr_ptr;
    // Low 32 bits of `mem_ptr` packed into 2 u16 cells. The upper 4 bytes are zero and
    // hardcoded in the memory bus interaction.
    T mem_ptr_limbs[MEM_PTR_NUM_LIMBS];
    MemoryReadAuxCols<T> mem_ptr_aux_cols;

    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    // 4 u16 cells holding the data word for the memory write. AS=`RV64_MEMORY_AS` is
    // u16-celled, so the bus consumes these cells directly without any byte-pair packing.
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
        // Pack low 4 bytes of `mem_ptr` into 2 u16 cells; the high 4 bytes are zero.
        auto mem_ptr_bytes = reinterpret_cast<uint8_t *>(&mem_ptr);
        uint32_t mem_ptr_limbs[MEM_PTR_NUM_LIMBS];
#pragma unroll
        for (size_t i = 0; i < MEM_PTR_NUM_LIMBS; i++) {
            mem_ptr_limbs[i] =
                uint32_t(mem_ptr_bytes[2 * i]) + 256u * uint32_t(mem_ptr_bytes[2 * i + 1]);
        }

        COL_WRITE_VALUE(row, Rv64HintStoreCols, is_single, is_single);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, is_buffer, !is_single);
        // `rem_words <= MAX_HINT_BUFFER_DWORDS < 2^16` fits in a single u16 cell.
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

            // Range check for `mem_ptr` (using pointer_max_bits): the high u16 cell of
            // the low 32 bits scaled by `1 << (32 - pointer_max_bits)` must fit in 16 bits.
            uint32_t mem_ptr_msl_lshift = 32u - (uint32_t)pointer_max_bits;
            uint32_t mem_ptr_high_u16 = record.mem_ptr >> 16;
            range_checker.add_count(mem_ptr_high_u16 << mem_ptr_msl_lshift, 16);

            // Range check for `rem_words` (using MAX_HINT_BUFFER_DWORDS_BITS): scaled by
            // `1 << (16 - MAX_HINT_BUFFER_DWORDS_BITS)` and range-checked to 16 bits.
            uint32_t rem_words_lshift = 16u - (uint32_t)MAX_HINT_BUFFER_DWORDS_BITS;
            range_checker.add_count(record.num_words << rem_words_lshift, 16);

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
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            packed_prev[i] = Fp(
                uint32_t(write.write_aux.prev_data[2 * i])
                + 256u * uint32_t(write.write_aux.prev_data[2 * i + 1])
            );
        }
        COL_WRITE_ARRAY(row, Rv64HintStoreCols, write_aux.prev_data, packed_prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64HintStoreCols, write_aux)),
            write.write_aux.prev_timestamp,
            timestamp + 2
        );

        // Pack 8 source bytes into 4 u16 cells; mirrors the Rust trace-fill loop.
        Fp packed_data[BLOCK_FE_WIDTH];
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            packed_data[i] =
                Fp(uint32_t(write.data[2 * i]) + 256u * uint32_t(write.data[2 * i + 1]));
        }
        COL_WRITE_ARRAY(row, Rv64HintStoreCols, data, packed_data);
    }
};

struct OffsetInfo {
    uint32_t record_offset;
    uint32_t local_idx;
};

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
