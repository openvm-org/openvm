#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/loadstore.cuh"

using namespace riscv;
using namespace program;

constexpr uint32_t LOADSTORE_SELECTOR_CASES = 30;
constexpr uint32_t LOADSTORE_SELECTOR_MAX_DEGREE = 2;
constexpr size_t LOADSTORE_SELECTOR_WIDTH = 7;

template <typename T, size_t NUM_CELLS> struct LoadStoreCoreCols {
    T selector[LOADSTORE_SELECTOR_WIDTH];
    /// we need to keep the degree of is_valid and is_load to 1
    T is_valid;
    T is_load;

    T read_data[NUM_CELLS];
    T prev_data[NUM_CELLS];
    /// write_data will be constrained against read_data and prev_data
    /// depending on the opcode and the shift amount
    T write_data[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadStoreCoreRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint8_t read_data[NUM_CELLS];
    // Note: `prev_data` can be a field, so we need to use u32
    uint32_t prev_data[NUM_CELLS];
};

enum Rv64LoadStoreOpcode {
    LOADD,
    LOADBU,
    LOADHU,
    LOADWU,
    STORED,
    STOREW,
    STOREH,
    STOREB,
};

__device__ __forceinline__ uint32_t instruction_case_from_opcode_shift(
    Rv64LoadStoreOpcode opcode,
    uint8_t shift
) {
    switch (opcode) {
    case LOADD:
        assert(shift == 0);
        return 0;
    case LOADWU:
        switch (shift) {
        case 0:
            return 1;
        case 4:
            return 2;
        }
        break;
    case LOADHU:
        switch (shift) {
        case 0:
            return 3;
        case 2:
            return 4;
        case 4:
            return 5;
        case 6:
            return 6;
        }
        break;
    case LOADBU:
        assert(shift < RV64_REGISTER_NUM_LIMBS);
        return 7 + shift;
    case STORED:
        assert(shift == 0);
        return 15;
    case STOREW:
        switch (shift) {
        case 0:
            return 16;
        case 4:
            return 17;
        }
        break;
    case STOREH:
        switch (shift) {
        case 0:
            return 18;
        case 2:
            return 19;
        case 4:
            return 20;
        case 6:
            return 21;
        }
        break;
    case STOREB:
        assert(shift < RV64_REGISTER_NUM_LIMBS);
        return 22 + shift;
    }

    assert(false);
    return 0;
}

template <size_t NUM_CELLS>
__device__ __forceinline__ void run_write_data(
    uint32_t (&write_data)[NUM_CELLS],
    Rv64LoadStoreOpcode opcode,
    const uint8_t (&read_data)[NUM_CELLS],
    const uint32_t (&prev_data)[NUM_CELLS],
    uint8_t shift
) {
    constexpr size_t WORD_WIDTH = NUM_CELLS / 2;
    constexpr size_t HALF_WIDTH = NUM_CELLS / 4;

    switch (opcode) {
    case LOADD:
    case STORED:
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            write_data[i] = read_data[i];
        }
        break;
    case LOADWU:
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            write_data[i] = i < WORD_WIDTH ? read_data[i + shift] : 0;
        }
        break;
    case LOADHU:
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            write_data[i] = i < HALF_WIDTH ? read_data[i + shift] : 0;
        }
        break;
    case LOADBU:
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            write_data[i] = i == 0 ? read_data[shift] : 0;
        }
        break;
    case STOREW:
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            if (i >= shift && i < shift + WORD_WIDTH) {
                write_data[i] = read_data[i - shift];
            } else {
                write_data[i] = prev_data[i];
            }
        }
        break;
    case STOREH:
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            if (i >= shift && i < shift + HALF_WIDTH) {
                write_data[i] = read_data[i - shift];
            } else {
                write_data[i] = prev_data[i];
            }
        }
        break;
    case STOREB:
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            write_data[i] = prev_data[i];
        }
        write_data[shift] = read_data[0];
        break;
    }
}

template <size_t NUM_CELLS> struct LoadStoreCore {

    template <typename T> using Cols = LoadStoreCoreCols<T, NUM_CELLS>;

    __device__ void fill_trace_row(RowSlice row, LoadStoreCoreRecord<NUM_CELLS> record) {
        Rv64LoadStoreOpcode opcode = static_cast<Rv64LoadStoreOpcode>(record.local_opcode);
        Encoder encoder(
            LOADSTORE_SELECTOR_CASES,
            LOADSTORE_SELECTOR_MAX_DEGREE,
            true
        );
        uint8_t shift = record.shift_amount;
        uint32_t write_data[NUM_CELLS] = {0};

        COL_WRITE_VALUE(row, Cols, is_valid, 1);
        COL_WRITE_VALUE(
            row,
            Cols,
            is_load,
            (opcode == LOADD || opcode == LOADWU || opcode == LOADHU || opcode == LOADBU)
        );
        encoder.write_flag_pt(
            row.slice_from(COL_INDEX(Cols, selector)),
            instruction_case_from_opcode_shift(opcode, shift)
        );
        COL_WRITE_ARRAY(row, Cols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, Cols, prev_data, record.prev_data);

        run_write_data(write_data, opcode, record.read_data, record.prev_data, shift);
        COL_WRITE_ARRAY(row, Cols, write_data, write_data);
    }
};

// [Adapter + Core] columns and record
template <typename T> struct Rv64LoadStoreCols {
    Rv64LoadStoreAdapterCols<T> adapter;
    LoadStoreCoreCols<T, RV64_REGISTER_NUM_LIMBS> core;
};

struct Rv64LoadStoreRecord {
    Rv64LoadStoreAdapterRecord adapter;
    LoadStoreCoreRecord<RV64_REGISTER_NUM_LIMBS> core;
};

__global__ void rv64_load_store_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64LoadStoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = LoadStoreCore<RV64_REGISTER_NUM_LIMBS>();
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadStoreCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv64LoadStoreCols<uint8_t>));
    }
}

extern "C" int _rv64_load_store_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv64LoadStoreCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    rv64_load_store_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
