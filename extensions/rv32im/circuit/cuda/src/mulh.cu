#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/mul.cuh"

using namespace riscv;

constexpr uint32_t NUM_LIMBS = RV32_REGISTER_NUM_LIMBS;
constexpr uint32_t LIMB_BITS = RV32_CELL_BITS;

template <typename T> struct MulHCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T a_mul[NUM_LIMBS];
    T b_ext;
    T c_ext;
    T opcode_mulh_flag;
    T opcode_mulhsu_flag;
    T opcode_mulhu_flag;
};

struct MulHCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

// Opcode mapping: MULH=0, MULHSU=1, MULHU=2
enum MulHOpcode { MULH = 0, MULHSU = 1, MULHU = 2 };

__device__ void run_mulh(
    MulHOpcode opcode,
    const uint32_t *x,
    const uint32_t *y,
    uint32_t *out_mulh,
    uint32_t *out_mul,
    uint32_t *out_carry,
    uint32_t &out_x_ext,
    uint32_t &out_y_ext
) {
#pragma unroll
    for (int i = 0; i < NUM_LIMBS; i++) {
        out_mul[i] = 0;
        out_carry[i] = 0;
        out_carry[NUM_LIMBS + i] = 0;
    }
#pragma unroll
    for (int i = 0; i < NUM_LIMBS; i++) {
        if (i > 0) {
            out_mul[i] = out_carry[i - 1];
        }
        for (int j = 0; j <= i; j++) {
            out_mul[i] += x[j] * y[i - j];
        }
        out_carry[i] = out_mul[i] >> LIMB_BITS;
        out_mul[i] %= (1u << LIMB_BITS);
    }

    out_x_ext =
        (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * (opcode == MULHU ? 0 : ((1u << LIMB_BITS) - 1));
    out_y_ext =
        (y[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * (opcode == MULH ? ((1u << LIMB_BITS) - 1) : 0);

    uint32_t x_prefix = 0;
    uint32_t y_prefix = 0;

#pragma unroll
    for (int i = 0; i < NUM_LIMBS; i++) {
        x_prefix += x[i];
        y_prefix += y[i];
        out_mulh[i] = out_carry[NUM_LIMBS + i - 1] + x_prefix * out_y_ext + y_prefix * out_x_ext;
#pragma unroll
        for (int j = i + 1; j < NUM_LIMBS; j++) {
            out_mulh[i] += x[j] * y[NUM_LIMBS + i - j];
        }
        out_carry[NUM_LIMBS + i] = out_mulh[i] >> LIMB_BITS;
        out_mulh[i] %= (1u << LIMB_BITS);
    }
}

struct MulHCore {
    RangeTupleChecker<2> range_tuple;
    BitwiseOperationLookup bitwise_lookup;

    __device__ MulHCore(
        uint32_t *range_tuple_ptr,
        uint32_t range_tuple_sizes[2],
        BitwiseOperationLookup bw
    )
        : range_tuple(range_tuple_ptr, range_tuple_sizes), bitwise_lookup(bw) {}

    __device__ void fill_trace_row(RowSlice row, MulHCoreRecord record) {
        MulHOpcode opcode = static_cast<MulHOpcode>(record.local_opcode);

        uint32_t b[NUM_LIMBS];
        uint32_t c[NUM_LIMBS];
#pragma unroll
        for (int i = 0; i < NUM_LIMBS; i++) {
            b[i] = static_cast<uint32_t>(record.b[i]);
            c[i] = static_cast<uint32_t>(record.c[i]);
        }

        uint32_t a[NUM_LIMBS];
        uint32_t a_mul[NUM_LIMBS];
        uint32_t carry[2 * NUM_LIMBS];
        uint32_t b_ext, c_ext;

        run_mulh(opcode, b, c, a, a_mul, carry, b_ext, c_ext);

#pragma unroll
        for (int i = 0; i < NUM_LIMBS; i++) {
            uint32_t aux[2] = {a_mul[i], carry[i]};
            range_tuple.add_count(aux);

            aux[0] = a[i];
            aux[1] = carry[NUM_LIMBS + i];
            range_tuple.add_count(aux);
        }

        if (opcode != MULHU) {
            uint32_t b_sign_mask = (b_ext == 0) ? 0 : (1u << (LIMB_BITS - 1));
            uint32_t c_sign_mask = (c_ext == 0) ? 0 : (1u << (LIMB_BITS - 1));

            bitwise_lookup.add_range(
                (b[NUM_LIMBS - 1] - b_sign_mask) << 1,
                (c[NUM_LIMBS - 1] - c_sign_mask) << (opcode == MULH)
            );
        }

        COL_WRITE_ARRAY(row, MulHCoreCols, a, a);
        COL_WRITE_ARRAY(row, MulHCoreCols, b, b);
        COL_WRITE_ARRAY(row, MulHCoreCols, c, c);
        COL_WRITE_ARRAY(row, MulHCoreCols, a_mul, a_mul);
        COL_WRITE_VALUE(row, MulHCoreCols, b_ext, b_ext);
        COL_WRITE_VALUE(row, MulHCoreCols, c_ext, c_ext);
        COL_WRITE_VALUE(row, MulHCoreCols, opcode_mulh_flag, opcode == MULH);
        COL_WRITE_VALUE(row, MulHCoreCols, opcode_mulhsu_flag, opcode == MULHSU);
        COL_WRITE_VALUE(row, MulHCoreCols, opcode_mulhu_flag, opcode == MULHU);
    }
};

template <typename T> struct MulHCols {
    Rv32MultAdapterCols<T> adapter;
    MulHCoreCols<T> core;
};

struct MulHRecord {
    Rv32MultAdapterRecord adapter;
    MulHCoreRecord core;
};

__global__ void mulh_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<MulHRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv32MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        MulHCore core(
            d_range_tuple_checker_ptr,
            (uint32_t[2]){range_tuple_checker_sizes.x, range_tuple_checker_sizes.y},
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(MulHCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(MulHCols<uint8_t>));
    }
}

extern "C" int _mulh_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<MulHRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(MulHCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    mulh_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        d_range_tuple_checker_ptr,
        range_tuple_checker_sizes,
        timestamp_max_bits
    );

    return cudaGetLastError();
}
