#include "constants.h"
#include "launcher.cuh"
#include "trace_access.h"
#include "adapters/alu_native_adapter.cuh"

using namespace riscv;

enum class FieldArithmeticOpcode : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3
};

struct FieldArithmeticCoreRecord {
    uint32_t b;
    uint32_t c;
    uint8_t  local_opcode;
};

template <typename T>
struct FieldArithmeticCoreCols {
    T a;
    T b;
    T c;
    T is_add;
    T is_sub;
    T is_mul;
    T is_div;
    T divisor_inv;
};

struct FieldArithmeticCore {
    __device__ void fill_trace_row(RowSlice row, FieldArithmeticCoreRecord rec) {
        const Fp b_val = Fp::fromRaw(rec.b);
        const Fp c_val = Fp::fromRaw(rec.c);
        const FieldArithmeticOpcode opcode = static_cast<FieldArithmeticOpcode>(rec.local_opcode);
        
        const bool is_add = (opcode == FieldArithmeticOpcode::ADD);
        const bool is_sub = (opcode == FieldArithmeticOpcode::SUB);
        const bool is_mul = (opcode == FieldArithmeticOpcode::MUL);
        const bool is_div = (opcode == FieldArithmeticOpcode::DIV);
        
        Fp result;
        switch (opcode) {
            case FieldArithmeticOpcode::ADD:
                result = b_val + c_val;
                break;
            case FieldArithmeticOpcode::SUB:
                result = b_val - c_val;
                break;
            case FieldArithmeticOpcode::MUL:
                result = b_val * c_val;
                break;
            case FieldArithmeticOpcode::DIV:
                result = b_val * inv(c_val);
                break;
            default:
                result = Fp::zero();
                break;
        }
        
        // Divisor inverse is only non-zero for division
        const Fp divisor_inv = is_div ? inv(c_val) : Fp::zero();
        
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, a, result);
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, b, b_val);
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, c, c_val);
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, is_add, Fp(is_add));
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, is_sub, Fp(is_sub));
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, is_mul, Fp(is_mul));
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, is_div, Fp(is_div));
        COL_WRITE_VALUE(row, FieldArithmeticCoreCols, divisor_inv, divisor_inv);
    }
};

template <typename T>
struct FieldArithmeticCols {
    AluNativeAdapterCols<T> adapter;
    FieldArithmeticCoreCols<T> core;
};

struct FieldArithmeticCompositeRecord {
    AluNativeAdapterRecord adapter;
    FieldArithmeticCoreRecord core;
};

__global__ void field_arithmetic_tracegen(
    Fp*        d_trace,
    size_t     height,
    uint8_t*   d_records,
    size_t     num_records,
    uint32_t*  d_range_checker,
    size_t     range_checker_bins
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    
    if (idx < num_records) {
        const auto rec = reinterpret_cast<FieldArithmeticCompositeRecord*>(d_records)[idx];
        
        AluNativeAdapter adapter(VariableRangeChecker(d_range_checker, range_checker_bins));
        adapter.fill_trace_row(row, rec.adapter);
        
        FieldArithmeticCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(FieldArithmeticCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(FieldArithmeticCols<uint8_t>));
    }
}

extern "C" int _field_arithmetic_tracegen(
    Fp*       d_trace,
    size_t    height,
    size_t    width,
    uint8_t*  d_records,
    size_t    record_len,
    uint32_t* d_range_checker,
    size_t    range_checker_bins
) {
    // Validate input parameters
    assert((height & (height - 1)) == 0);  // height must be power of 2
    assert(width == sizeof(FieldArithmeticCols<uint8_t>));
    
    const size_t num_records = record_len / sizeof(FieldArithmeticCompositeRecord);
    const auto [grid, block] = kernel_launch_params(height);
    
    field_arithmetic_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker,
        range_checker_bins
    );
    
    return cudaGetLastError();
}