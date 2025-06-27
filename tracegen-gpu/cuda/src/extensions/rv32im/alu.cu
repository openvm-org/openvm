#include "adapters/alu.cuh"
#include "constants.h"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

struct Rv32BaseAluCoreRecord {
    uint8_t b[RV32_REGISTER_NUM_LIMBS];
    uint8_t c[RV32_REGISTER_NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T>
struct Rv32BaseAluCoreCols {
    T a[RV32_REGISTER_NUM_LIMBS];
    T b[RV32_REGISTER_NUM_LIMBS];
    T c[RV32_REGISTER_NUM_LIMBS];
    T opcode_add_flag;
    T opcode_sub_flag;
    T opcode_xor_flag;
    T opcode_or_flag;
    T opcode_and_flag;
};

__device__ void run_add(const uint8_t* x, const uint8_t* y, uint8_t* out, uint8_t* carry) {
    for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        uint32_t res = (i > 0) ? carry[i - 1] : 0;
        res += static_cast<uint32_t>(x[i]) + static_cast<uint32_t>(y[i]);
        carry[i] = res >> RV32_CELL_BITS;
        out[i] = static_cast<uint8_t>(res & ((1u << RV32_CELL_BITS) - 1));
    }
}

__device__ void run_sub(const uint8_t* x, const uint8_t* y, uint8_t* out, uint8_t* carry) {
    for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        uint32_t rhs = static_cast<uint32_t>(y[i]) + ((i > 0) ? carry[i - 1] : 0);
        if (static_cast<uint32_t>(x[i]) >= rhs) {
            out[i] = static_cast<uint8_t>(static_cast<uint32_t>(x[i]) - rhs);
            carry[i] = 0;
        } else {
            uint32_t wrap = (static_cast<uint32_t>(1u << RV32_CELL_BITS) + static_cast<uint32_t>(x[i]) - rhs);
            out[i] = static_cast<uint8_t>(wrap);
            carry[i] = 1;
        }
    }
}

__device__ void run_xor(const uint8_t* x, const uint8_t* y, uint8_t* out) {
    for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        out[i] = x[i] ^ y[i];
    }
}

__device__ void run_or(const uint8_t* x, const uint8_t* y, uint8_t* out) {
    for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        out[i] = x[i] | y[i];
    }
}

__device__ void run_and(const uint8_t* x, const uint8_t* y, uint8_t* out) {
    for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        out[i] = x[i] & y[i];
    }
}

struct Rv32BaseAluCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ Rv32BaseAluCore(BitwiseOperationLookup lookup)
        : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, Rv32BaseAluCoreRecord record) {
        uint8_t a[RV32_REGISTER_NUM_LIMBS];
        uint8_t carry_buf[RV32_REGISTER_NUM_LIMBS];

        switch (record.local_opcode) {
            case 0: 
                run_add(record.b, record.c, a, carry_buf);
                break;
            case 1:
                run_sub(record.b, record.c, a, carry_buf);
                break;
            case 2:
                run_xor(record.b, record.c, a);
                break;
            case 3:
                run_or(record.b, record.c, a);
                break;
            case 4:
                run_and(record.b, record.c, a);
                break;
            default:
#pragma unroll
                for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
                    a[i] = 0;
                    carry_buf[i] = 0;
                }
        }

        COL_WRITE_ARRAY(row, Rv32BaseAluCoreCols, b, record.b);
        COL_WRITE_ARRAY(row, Rv32BaseAluCoreCols, c, record.c);
        COL_WRITE_ARRAY(row, Rv32BaseAluCoreCols, a, a);
        COL_WRITE_VALUE(row, Rv32BaseAluCoreCols, opcode_add_flag, record.local_opcode == 0);
        COL_WRITE_VALUE(row, Rv32BaseAluCoreCols, opcode_sub_flag, record.local_opcode == 1);
        COL_WRITE_VALUE(row, Rv32BaseAluCoreCols, opcode_xor_flag, record.local_opcode == 2);
        COL_WRITE_VALUE(row, Rv32BaseAluCoreCols, opcode_or_flag, record.local_opcode == 3);
        COL_WRITE_VALUE(row, Rv32BaseAluCoreCols, opcode_and_flag, record.local_opcode == 4);

#pragma unroll
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
            if (record.local_opcode == 0 || record.local_opcode == 1) {
                bitwise_lookup.add_range(a[i], a[i]);
            } else {
                bitwise_lookup.add_xor(record.b[i], record.c[i]);
            }
        }
    }
};

template <typename T>
struct Rv32BaseAluCols {
    Rv32BaseAluAdapterCols<T> adapter;
    Rv32BaseAluCoreCols<T> core;
};

struct Rv32BaseAluRecord {
    Rv32BaseAluAdapterRecord adapter;
    Rv32BaseAluCoreRecord    core;
};

__global__ void alu_tracegen(
    Fp*       d_trace,
    size_t    height,
    uint8_t*  d_records,
    size_t    num_records,
    uint32_t* d_range_checker_ptr,
    size_t    range_checker_bins,
    uint32_t* d_bitwise_lookup_ptr,
    size_t    bitwise_num_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<Rv32BaseAluRecord*>(d_records)[idx];

        Rv32BaseAluAdapter adapter(VariableRangeChecker(d_range_checker_ptr, range_checker_bins));
        adapter.fill_trace_row(row, rec.adapter);

        Rv32BaseAluCore core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv32BaseAluCols, core)),
            rec.core
        );
    } else {
        row.fill_zero(0, sizeof(Rv32BaseAluCols<uint8_t>));
    }
}

extern "C" int _alu_tracegen(
    Fp*       d_trace,
    size_t    height,
    size_t    width,
    uint8_t*  d_records,
    size_t    record_len,
    uint32_t* d_range_checker_ptr,
    size_t    range_checker_bins,
    uint32_t* d_bitwise_lookup_ptr,
    size_t    bitwise_num_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(Rv32BaseAluRecord) >= record_len);
    assert(width == sizeof(Rv32BaseAluCols<uint8_t>));
    size_t num_records = record_len / sizeof(Rv32BaseAluRecord);
    auto [grid, block] = kernel_launch_params(height);
    alu_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits
    );
    return cudaGetLastError();
} 