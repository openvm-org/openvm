#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/is_equal.cuh"
#include "primitives/trace_access.h"
#include "rv32-adapters/eq_mod.cuh"

using namespace riscv;

template <size_t READ_LIMBS> struct ModularIsEqualCoreRecord {
    uint8_t is_setup;
    uint8_t b[READ_LIMBS];
    uint8_t c[READ_LIMBS];
};

template <typename T, size_t READ_LIMBS> struct ModularIsEqualCoreCols {
    T is_valid;
    T is_setup;
    T b[READ_LIMBS];
    T c[READ_LIMBS];
    T cmp_result;
    T eq_marker[READ_LIMBS];
    T lt_marker[READ_LIMBS];
    T b_lt_diff;
    T c_lt_diff;
    T c_lt_mark;
};

template <size_t READ_LIMBS> struct ModularIsEqualCore {
    const uint8_t *modulus;
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = ModularIsEqualCoreCols<T, READ_LIMBS>;

    __device__ ModularIsEqualCore(const uint8_t *mod, BitwiseOperationLookup lookup)
        : modulus(mod), bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, ModularIsEqualCoreRecord<READ_LIMBS> record) {
        COL_WRITE_VALUE(row, Cols, is_valid, Fp::one());
        COL_WRITE_VALUE(row, Cols, is_setup, Fp(record.is_setup));

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);

        RowSlice b_slice = row.slice_from(COL_INDEX(Cols, b));
        RowSlice c_slice = row.slice_from(COL_INDEX(Cols, c));
        RowSlice eq_marker_slice = row.slice_from(COL_INDEX(Cols, eq_marker));
        Fp *cmp_result_ptr = &row[COL_INDEX(Cols, cmp_result)];
        IsEqualArray::generate_subrow(
            READ_LIMBS, b_slice, c_slice, eq_marker_slice, cmp_result_ptr
        );

        int b_diff = READ_LIMBS; // essentially means "no differing limb"
        int c_diff = READ_LIMBS;
        for (int i = READ_LIMBS - 1; i >= 0; i--) {
            if (record.b[i] != modulus[i]) {
                b_diff = i;
                break;
            }
        }
        for (int i = READ_LIMBS - 1; i >= 0; i--) {
            if (record.c[i] != modulus[i]) {
                c_diff = i;
                break;
            }
        }

        Fp mark = (b_diff == c_diff) ? Fp::one() : Fp(2);
        COL_WRITE_VALUE(row, Cols, c_lt_mark, mark);

        Fp c_diff_val = Fp(modulus[c_diff % READ_LIMBS] - record.c[c_diff % READ_LIMBS]);
        COL_WRITE_VALUE(row, Cols, c_lt_diff, c_diff_val);

        if (record.is_setup == 0) {
            Fp b_diff_val = Fp(modulus[b_diff % READ_LIMBS] - record.b[b_diff % READ_LIMBS]);
            COL_WRITE_VALUE(row, Cols, b_lt_diff, b_diff_val);

            uint32_t b_lt_diff = modulus[b_diff] - record.b[b_diff];
            uint32_t c_lt_diff = modulus[c_diff] - record.c[c_diff];
            bitwise_lookup.add_range(b_lt_diff - 1, c_lt_diff - 1);
        } else {
            COL_WRITE_VALUE(row, Cols, b_lt_diff, Fp::zero());
        }

        for (uint32_t i = 0; i < READ_LIMBS; i++) {
            Fp v = Fp::zero();
            if (b_diff < READ_LIMBS && (int)i == b_diff)
                v = Fp::one();
            else if (c_diff < READ_LIMBS && (int)i == c_diff)
                v = mark;
            COL_WRITE_VALUE(row, Cols, lt_marker[i], v);
        }
    }
};

template <typename T, size_t NUM_READS, size_t NUM_LANES, size_t LANE_SIZE, size_t TOTAL_LIMBS>
struct ModularIsEqualCols {
    Rv32IsEqualModAdapterCols<T, NUM_READS, NUM_LANES, LANE_SIZE, TOTAL_LIMBS> adapter;
    ModularIsEqualCoreCols<T, TOTAL_LIMBS> core;
};

template <size_t NUM_READS, size_t NUM_LANES, size_t LANE_SIZE, size_t TOTAL_LIMBS>
struct ModularIsEqualRecord {
    Rv32IsEqualModAdapterRecord<NUM_READS, NUM_LANES, LANE_SIZE, TOTAL_LIMBS> adapter;
    ModularIsEqualCoreRecord<TOTAL_LIMBS> core;
};

template <size_t NUM_READS, size_t NUM_LANES, size_t LANE_SIZE, size_t TOTAL_LIMBS>
__global__ void modular_is_equal_tracegen_kernel(
    Fp *d_trace,
    size_t height,
    size_t width,
    const uint8_t *d_records,
    size_t num_records,
    const uint8_t *d_modulus,
    const uint32_t *d_range_ctr,
    size_t range_bins,
    const uint32_t *d_bitwise_lut,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < num_records) {
        using RecordType = ModularIsEqualRecord<NUM_READS, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>;
        const RecordType &rec = reinterpret_cast<const RecordType *>(d_records)[idx];

        VariableRangeChecker rc(const_cast<uint32_t *>(d_range_ctr), range_bins);
        BitwiseOperationLookup bl(const_cast<uint32_t *>(d_bitwise_lut), bitwise_num_bits);
        Rv32IsEqualModAdapter<NUM_READS, NUM_LANES, LANE_SIZE, TOTAL_LIMBS> adapter(
            rc, bl, pointer_max_bits, timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        constexpr size_t adapter_cols = sizeof(
            Rv32IsEqualModAdapterCols<uint8_t, NUM_READS, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
        );
        RowSlice core_slice = row.slice_from(adapter_cols);

        ModularIsEqualCore<TOTAL_LIMBS> core(d_modulus, bl);
        core.fill_trace_row(core_slice, rec.core);
    } else if (idx < height) {
        row.fill_zero(0, width);
    }
}

extern "C" int _modular_is_equal_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    const uint8_t *d_records,
    size_t record_len,
    const uint8_t *d_modulus,
    size_t total_limbs,
    size_t num_lanes,
    size_t lane_size,
    const uint32_t *d_range_ctr,
    size_t range_bins,
    const uint32_t *d_bitwise_lut,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height, 256);

    constexpr size_t NUM_READS = 2;

    if (num_lanes == 1 && lane_size == 32 && total_limbs == 32) {
        using RecordType = ModularIsEqualRecord<NUM_READS, 1, 32, 32>;
        using ColsType = ModularIsEqualCols<uint8_t, NUM_READS, 1, 32, 32>;

        assert(width == sizeof(ColsType));
        size_t num_records = record_len / sizeof(RecordType);

        modular_is_equal_tracegen_kernel<NUM_READS, 1, 32, 32><<<grid, block>>>(
            d_trace,
            height,
            width,
            d_records,
            num_records,
            d_modulus,
            d_range_ctr,
            range_bins,
            d_bitwise_lut,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits
        );
    } else if (num_lanes == 3 && lane_size == 16 && total_limbs == 48) {
        using RecordType = ModularIsEqualRecord<NUM_READS, 3, 16, 48>;
        using ColsType = ModularIsEqualCols<uint8_t, NUM_READS, 3, 16, 48>;

        assert(width == sizeof(ColsType));
        size_t num_records = record_len / sizeof(RecordType);

        modular_is_equal_tracegen_kernel<NUM_READS, 3, 16, 48><<<grid, block>>>(
            d_trace,
            height,
            width,
            d_records,
            num_records,
            d_modulus,
            d_range_ctr,
            range_bins,
            d_bitwise_lut,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits
        );
    } else {
        return cudaErrorInvalidValue;
    }

    return CHECK_KERNEL();
}