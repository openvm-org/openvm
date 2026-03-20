#include "fp.h"
#include "poseidon2-air/columns.cuh"
#include "poseidon2-air/params.cuh"
#include "poseidon2-air/tracegen.cuh"
#include "primitives/encoder.cuh"
#include "primitives/trace_access.h"
#include "types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

inline constexpr size_t SBOX_REGS = Poseidon2ParamsS1::SBOX_REGS;
inline constexpr size_t SBOX_DEGREE = Poseidon2DefaultParams::SBOX_DEGREE;
inline constexpr size_t HALF_FULL_ROUNDS = Poseidon2DefaultParams::HALF_FULL_ROUNDS;
inline constexpr size_t PARTIAL_ROUNDS = Poseidon2DefaultParams::PARTIAL_ROUNDS;
inline constexpr size_t NUM_FLAGS = 4;

template <typename T> struct DagCommitCols {
    poseidon2::Poseidon2SubCols<T, WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
        inner;
    T flags[NUM_FLAGS];
    T is_constraint;
};

__device__ __forceinline__ void write_dag_commit_poseidon2(
    RowSlice row,
    Fp *poseidon2_input,
    bool is_constraint
) {
    using Poseidon2Row =
        poseidon2::Poseidon2Row<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

    Poseidon2Row perm(row);
    RowSlice state(poseidon2_input, 1);
    poseidon2::generate_trace_row_for_perm(perm, state);

    row.fill_zero(COL_INDEX(DagCommitCols, flags), NUM_FLAGS);
    COL_WRITE_VALUE(row, DagCommitCols, is_constraint, is_constraint ? Fp::one() : Fp::zero());
}

__device__ __forceinline__ void write_dag_commit_flags(
    RowSlice row,
    Encoder &encoder,
    uint32_t node_kind
) {
    assert(encoder.k == NUM_FLAGS);
    encoder.write_flag_pt(row.slice_from(COL_INDEX(DagCommitCols, flags)), node_kind);
}
