#include "constants.h"
#include "histogram.cuh"
#include "launcher.cuh"
#include "rv32_adapters/heap.cuh"
#include "rv32_adapters/heap_branch.cuh"
#include "rv32im/cores/alu.cuh"
#include "rv32im/cores/beq.cuh"
#include "rv32im/cores/branch_less_than.cuh"
#include "rv32im/cores/less_than.cuh"
#include "rv32im/cores/multiplication.cuh"
#include "rv32im/cores/shift.cuh"
#include "trace_access.h"

using namespace riscv;

constexpr size_t INT256_NUM_LIMBS = 32;

using BaseAlu256CoreRecord = BaseAluCoreRecord<32>;
using BaseAlu256Core = BaseAluCore<32>;
template <typename T> using BaseAlu256CoreCols = BaseAluCoreCols<T, 32>;

using BranchEqual256Core = BranchEqualCore<32>;
template <typename T> using BranchEqual256CoreCols = BranchEqualCoreCols<T, 32>;
using BranchEqual256CoreRecord = BranchEqualCoreRecord<32>;

using LessThan256CoreRecord = LessThanCoreRecord<32>;
using LessThan256Core = LessThanCore<32>;
template <typename T> using LessThan256CoreCols = LessThanCoreCols<T, 32>;

using Multiplication256CoreRecord = MultiplicationCoreRecord<32>;
using Multiplication256Core = MultiplicationCore<32>;
template <typename T> using Multiplication256CoreCols = MultiplicationCoreCols<T, 32>;

using Shift256CoreRecord = ShiftCoreRecord<32>;
using Shift256Core = ShiftCore<32>;
template <typename T> using Shift256CoreCols = ShiftCoreCols<T, 32>;

using BranchLessThan256CoreRecord = BranchLessThanCoreRecord<32>;
using BranchLessThan256Core = BranchLessThanCore<32>;
template <typename T> using BranchLessThan256CoreCols = BranchLessThanCoreCols<T, 32>;

// Heap adapter instantiation for 256-bit operations
// NUM_READS = 2, READ_SIZE = INT256_NUM_LIMBS (32 bytes), WRITE_SIZE = INT256_NUM_LIMBS (32 bytes)
// BLOCKS_PER_READ = 1, BLOCKS_PER_WRITE = 1
using Rv32HeapAdapterStep256 = Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

template <typename T> struct BaseAlu256Cols {
    Rv32HeapAdapterCols<T, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    BaseAlu256CoreCols<T> core;
};

struct BaseAlu256Record {
    Rv32HeapAdapterRecord<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    BaseAlu256CoreRecord core;
};

__global__ void alu256_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<BaseAlu256Record *>(d_records)[idx];

        Rv32HeapAdapterStep256 adapter(
            32,
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        BaseAlu256Core core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(BaseAlu256Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(BaseAlu256Cols<uint8_t>));
    }
}

extern "C" int _alu256_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(BaseAlu256Record) >= record_len);
    assert(width == sizeof(BaseAlu256Cols<uint8_t>));
    size_t num_records = record_len / sizeof(BaseAlu256Record);

    auto [grid, block] = kernel_launch_params(height);
    alu256_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}

// Heap branch adapter instantiation for 256-bit operations
// NUM_READS = 2, READ_SIZE = INT256_NUM_LIMBS (32 bytes)
using Rv32HeapBranchAdapter256 = Rv32HeapBranchAdapter<2, INT256_NUM_LIMBS>;

template <typename T> struct BranchEqual256Cols {
    Rv32HeapBranchAdapterCols<T, 2, INT256_NUM_LIMBS> adapter;
    BranchEqual256CoreCols<T> core;
};

struct BranchEqual256Record {
    Rv32HeapBranchAdapterRecord<2> adapter;
    BranchEqual256CoreRecord core;
};

__global__ void branch_equal256_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<BranchEqual256Record *>(d_records)[idx];

        Rv32HeapBranchAdapter256 adapter(
            32,
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        BranchEqual256Core core;
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchEqual256Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(BranchEqual256Cols<uint8_t>));
    }
}

extern "C" int _branch_equal256_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(BranchEqual256Record) >= record_len);
    assert(width == sizeof(BranchEqual256Cols<uint8_t>));
    size_t num_records = record_len / sizeof(BranchEqual256Record);

    auto [grid, block] = kernel_launch_params(height);
    branch_equal256_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}

template <typename T> struct LessThan256Cols {
    Rv32HeapAdapterCols<T, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    LessThan256CoreCols<T> core;
};

struct LessThan256Record {
    Rv32HeapAdapterRecord<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    LessThan256CoreRecord core;
};

__global__ void less_than256_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<LessThan256Record *>(d_records)[idx];

        Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter(
            32,
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        LessThan256Core core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(LessThan256Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(LessThan256Cols<uint8_t>));
    }
}

extern "C" int _less_than256_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(LessThan256Record) >= record_len);
    assert(width == sizeof(LessThan256Cols<uint8_t>));
    size_t num_records = record_len / sizeof(LessThan256Record);

    auto [grid, block] = kernel_launch_params(height);
    less_than256_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}

template <typename T> struct BranchLessThan256Cols {
    Rv32HeapBranchAdapterCols<T, 2, INT256_NUM_LIMBS> adapter;
    BranchLessThan256CoreCols<T> core;
};

struct BranchLessThan256Record {
    Rv32HeapBranchAdapterRecord<2> adapter;
    BranchLessThan256CoreRecord core;
};

__global__ void branch_less_than256_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<BranchLessThan256Record *>(d_records)[idx];

        Rv32HeapBranchAdapter256 adapter(
            32,
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        BranchLessThan256Core core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchLessThan256Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(BranchLessThan256Cols<uint8_t>));
    }
}

extern "C" int _branch_less_than256_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(BranchLessThan256Record) >= record_len);
    assert(width == sizeof(BranchLessThan256Cols<uint8_t>));
    size_t num_records = record_len / sizeof(BranchLessThan256Record);

    auto [grid, block] = kernel_launch_params(height);
    branch_less_than256_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}

template <typename T> struct Shift256Cols {
    Rv32HeapAdapterCols<T, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    Shift256CoreCols<T> core;
};

struct Shift256Record {
    Rv32HeapAdapterRecord<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    Shift256CoreRecord core;
};

__global__ void shift256_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<Shift256Record *>(d_records)[idx];

        Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter(
            32,
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Shift256Core core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Shift256Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Shift256Cols<uint8_t>));
    }
}

extern "C" int _shift256_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(Shift256Record) >= record_len);
    assert(width == sizeof(Shift256Cols<uint8_t>));
    size_t num_records = record_len / sizeof(Shift256Record);

    auto [grid, block] = kernel_launch_params(height);
    shift256_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}

template <typename T> struct Multiplication256Cols {
    Rv32HeapAdapterCols<T, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    Multiplication256CoreCols<T> core;
};

struct Multiplication256Record {
    Rv32HeapAdapterRecord<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter;
    Multiplication256CoreRecord core;
};

__global__ void multiplication256_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<Multiplication256Record *>(d_records)[idx];

        Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS> adapter(
            32,
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Multiplication256Core core(range_tuple_checker);
        core.fill_trace_row(row.slice_from(COL_INDEX(Multiplication256Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Multiplication256Cols<uint8_t>));
    }
}

extern "C" int _multiplication256_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(Multiplication256Record) >= record_len);
    assert(width == sizeof(Multiplication256Cols<uint8_t>));
    size_t num_records = record_len / sizeof(Multiplication256Record);

    auto [grid, block] = kernel_launch_params(height);
    multiplication256_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return cudaGetLastError();
}
