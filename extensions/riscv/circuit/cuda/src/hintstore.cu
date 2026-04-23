#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

using namespace riscv;
using namespace program;
using hintstore::MAX_HINT_BUFFER_DWORDS;
using hintstore::MAX_HINT_BUFFER_DWORDS_BITS;

constexpr size_t REM_WORDS_NUM_LIMBS = 2;

template <typename T> struct Rv64HintStoreCols {
    // common
    T is_single;
    T is_buffer;

    // Low bytes of the 8-byte RV64 register that holds `rem_words`; the upper 6 bytes are
    // known to be zero and are hardcoded in the memory bus interaction.
    T rem_words_limbs[REM_WORDS_NUM_LIMBS];

    ExecutionState<T> from_state;
    T mem_ptr_ptr;
    T mem_ptr_limbs[RV64_WORD_NUM_LIMBS];
    MemoryReadAuxCols<T> mem_ptr_aux_cols;

    MemoryWriteAuxCols<T, RV64_REGISTER_NUM_LIMBS> write_aux;
    T data[RV64_REGISTER_NUM_LIMBS];

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
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64HintStore(
        BitwiseOperationLookup bitwise_lookup,
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : bitwise_lookup(bitwise_lookup), pointer_max_bits(pointer_max_bits),
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
        auto rem_words_limbs = reinterpret_cast<uint8_t *>(&rem_words);
        auto mem_ptr_limbs = reinterpret_cast<uint8_t *>(&mem_ptr);

        COL_WRITE_VALUE(row, Rv64HintStoreCols, is_single, is_single);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, is_buffer, !is_single);
        COL_WRITE_ARRAY(row, Rv64HintStoreCols, rem_words_limbs, rem_words_limbs);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, from_state.timestamp, timestamp);
        COL_WRITE_VALUE(row, Rv64HintStoreCols, mem_ptr_ptr, record.mem_ptr_ptr);
        COL_WRITE_ARRAY(row, Rv64HintStoreCols, mem_ptr_limbs, mem_ptr_limbs);

        if (local_idx == 0) {
            // Range check for mem_ptr (using pointer_max_bits). `mem_ptr` is a 4-limb value;
            // the range check packs the most-significant of those 4 limbs into a single byte.
            uint32_t msl_rshift = (RV64_WORD_NUM_LIMBS - 1) * RV64_CELL_BITS;
            uint32_t msl_lshift = RV64_WORD_NUM_LIMBS * RV64_CELL_BITS - pointer_max_bits;

            // Range check for num_words (using MAX_HINT_BUFFER_DWORDS_BITS).
            uint32_t rem_words_msl_lshift =
                REM_WORDS_NUM_LIMBS * RV64_CELL_BITS - MAX_HINT_BUFFER_DWORDS_BITS;

            // Combined range check for mem_ptr and num_words
            bitwise_lookup.add_range(
                (record.mem_ptr >> msl_rshift) << msl_lshift,
                ((record.num_words >> (RV64_CELL_BITS * (REM_WORDS_NUM_LIMBS - 1))) & 0xFF)
                    << rem_words_msl_lshift
            );
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

        COL_WRITE_ARRAY(row, Rv64HintStoreCols, write_aux.prev_data, write.write_aux.prev_data);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64HintStoreCols, write_aux)),
            write.write_aux.prev_timestamp,
            timestamp + 2
        );

        COL_WRITE_ARRAY(row, Rv64HintStoreCols, data, write.data);
#pragma unroll
        for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i += 2) {
            bitwise_lookup.add_range(write.data[i], write.data[i + 1]);
        }
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
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
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
            BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits),
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
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(height == 0 || (height & (height - 1)) == 0);
    assert(height >= rows_used);
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
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
