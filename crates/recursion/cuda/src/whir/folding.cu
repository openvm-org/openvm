#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <stdio.h>

template <typename T> struct WhirFoldingCols {
    T is_valid;
    T proof_idx;
    T whir_round;
    T query_idx;
    T is_root;
    T coset_shift;
    T coset_idx;
    T height;
    T twiddle;
    T coset_size;
    T z_final;
    T value[D_EF];
    T left_value[D_EF];
    T right_value[D_EF];
    T y_final[D_EF];
    T alpha[D_EF];
};

struct FoldRecord {
    uint32_t whir_round;
    uint32_t query_idx;
    uint32_t coset_idx;
    uint32_t height;
    uint32_t coset_size;
    Fp coset_shift;
    Fp twiddle;
    Fp z_final;
    FpExt value;
    FpExt left_value;
    FpExt right_value;
    FpExt y_final;
    FpExt alpha;
};

__global__ void whir_folding_tracegen_kernel(
    Fp *trace,
    uint32_t num_valid_rows,
    uint32_t height,
    const FoldRecord *records,
    uint32_t num_rounds,
    uint32_t num_queries,
    uint32_t k_whir
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height) {
        return;
    }

    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_valid_rows) {
        row.fill_zero(0, sizeof(WhirFoldingCols<uint8_t>));
        return;
    }

    const FoldRecord record = records[row_idx];

    const uint32_t rows_per_proof =
        num_rounds * num_queries * ((1 << k_whir) - 1);
    assert(rows_per_proof > 0);
    const uint32_t proof_idx = row_idx / rows_per_proof;

    COL_WRITE_VALUE(row, WhirFoldingCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, WhirFoldingCols, proof_idx, Fp(proof_idx));
    COL_WRITE_VALUE(row, WhirFoldingCols, whir_round, Fp(record.whir_round));
    COL_WRITE_VALUE(row, WhirFoldingCols, query_idx, Fp(record.query_idx));
    COL_WRITE_VALUE(
        row,
        WhirFoldingCols,
        is_root,
        record.coset_size == 1 ? Fp::one() : Fp::zero()
    );
    COL_WRITE_VALUE(row, WhirFoldingCols, coset_shift, record.coset_shift);
    COL_WRITE_VALUE(row, WhirFoldingCols, coset_idx, Fp(record.coset_idx));
    COL_WRITE_VALUE(row, WhirFoldingCols, height, Fp(record.height));
    COL_WRITE_VALUE(row, WhirFoldingCols, twiddle, record.twiddle);
    COL_WRITE_VALUE(row, WhirFoldingCols, coset_size, Fp(record.coset_size));
    COL_WRITE_ARRAY(row, WhirFoldingCols, value, record.value.elems);
    COL_WRITE_ARRAY(row, WhirFoldingCols, left_value, record.left_value.elems);
    COL_WRITE_ARRAY(row, WhirFoldingCols, right_value, record.right_value.elems);
    COL_WRITE_VALUE(row, WhirFoldingCols, z_final, record.z_final);
    COL_WRITE_ARRAY(row, WhirFoldingCols, y_final, record.y_final.elems);
    COL_WRITE_ARRAY(row, WhirFoldingCols, alpha, record.alpha.elems);
}

extern "C" int _whir_folding_tracegen(
    Fp *trace,
    uint32_t num_valid_rows,
    uint32_t height,
    const FoldRecord *records,
    uint32_t num_rounds,
    uint32_t num_queries,
    uint32_t k_whir
) {
    if (height == 0) {
        return 0;
    }

    auto [grid, block] = kernel_launch_params(height);
    whir_folding_tracegen_kernel<<<grid, block>>>(
        trace,
        num_valid_rows,
        height,
        records,
        num_rounds,
        num_queries,
        k_whir
    );
    return CHECK_KERNEL();
}
