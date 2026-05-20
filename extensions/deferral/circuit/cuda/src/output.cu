#include <cassert>
#include <cstddef>
#include <cstdint>

#include "canonicity.cuh"
#include "def_poseidon2_buffer.cuh"
#include "def_types.h"
#include "fp.h"
#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/fp_array.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace deferral;
using namespace canonicity;
using namespace lookup;

template <typename T> using MemoryWriteAuxColsDef = MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>;

__device__ __forceinline__ void write_canonicity_aux(
    RowSlice row,
    size_t base_offset,
    size_t aux_idx,
    const CanonicityAuxCols<Fp> &aux
) {
    constexpr size_t aux_stride = sizeof(CanonicityAuxCols<uint8_t>);
    RowSlice aux_row = row.slice_from(base_offset + aux_idx * aux_stride);
    COL_WRITE_ARRAY(aux_row, CanonicityAuxCols, diff_marker, aux.diff_marker);
    COL_WRITE_VALUE(aux_row, CanonicityAuxCols, diff_val, aux.diff_val);
}

template <typename T> struct DeferralOutputCols {
    // Indicates the status of this row, i.e. if it is valid and where it is in a
    // section of rows that correspond to a single opcode invocation
    T is_valid;
    T is_first;
    T is_last;
    T section_idx;

    // Initial execution state + instruction operands
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs_ptr;
    T deferral_idx;

    // Low 32 bits of heap pointers, packed as u16 cells.
    T rd_val[RV64_PTR_U16S];
    T rs_val[RV64_PTR_U16S];
    MemoryReadAuxCols<T> rd_aux;
    MemoryReadAuxCols<T> rs_aux;

    // First row reads [output_commit || output_len_le] from heap as u16 cells.
    T output_commit[COMMIT_NUM_U16S];
    T output_len[F_NUM_U16S];
    MemoryReadAuxCols<T> output_commit_and_len_aux[OUTPUT_TOTAL_MEMORY_OPS];

    // Auxiliary columns to ensure canonicity of output_commit cells.
    CanonicityAuxCols<T> output_commit_lt_aux[DIGEST_SIZE];

    // First row sponge input is [deferral_idx, output_len, 0, ...]. Later rows absorb and write
    // SPONGE_BYTES_PER_ROW bytes as DIGEST_SIZE u16 cells.
    T sponge_inputs[DIGEST_SIZE];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_bytes_aux[SPONGE_ROW_MEMORY_OPS];

    // Capacity of the permutation of write_bytes and the previous row's capacity on
    // non-last rows, compression on the last row.
    T poseidon2_res[DIGEST_SIZE];
};
