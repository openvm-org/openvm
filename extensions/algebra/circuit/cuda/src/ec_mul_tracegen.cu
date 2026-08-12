// Trace generation for the `EC_MUL` chip. The per-row and per-instruction bodies are in
// `algebra/ec_mul_tracegen.cuh`; this file supplies the kernels and the launch policy.
//
// Like the `EC_MUL` projection gather, this lives in the algebra crate because that is the crate
// compiling CUDA for the extension stack; the ECC circuit crate has no CUDA build of its own.
#include "algebra/ec_mul_tracegen.cuh"
#include "algebra/ec_mul_projective.cuh"
#include "launcher.cuh"

template <uint32_t K, size_t BLOCKS>
static __global__ void ec_mul_projective_prepare_pass(
    const EcMulTraceInput<BLOCKS> *projection,
    size_t num_instructions,
    const uint32_t *blob,
    uint32_t *projective,
    uint32_t *error
) {
    if (*error != 0) return;
    size_t instruction = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (instruction >= num_instructions || projection[instruction].is_setup != 0) return;
    FieldExprProg s;
    load_prog(blob, s);
    uint32_t *rows = projective + instruction * EC_MUL_PROJECTIVE_INSTRUCTION_WORDS<K>;
    ec_mul_projective_build_projective<K>(s, projection[instruction], rows, error);
}

template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __global__ void ec_mul_projective_batch_invert_pass(
    const EcMulTraceInput<BLOCKS> *projection,
    size_t num_instructions,
    const uint32_t *blob,
    uint32_t *projective,
    uint32_t *error
) {
    if (*error != 0) return;
    size_t instruction = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (instruction >= num_instructions || projection[instruction].is_setup != 0) return;
    FieldExprProg s;
    load_prog(blob, s);
    uint32_t *rows = projective + instruction * EC_MUL_PROJECTIVE_INSTRUCTION_WORDS<K>;
    ec_mul_projective_batch_invert<K>(s, rows, error);
}

template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __global__ void ec_mul_projective_materialize_pass(
    const EcMulTraceInput<BLOCKS> *projection,
    size_t num_instructions,
    const uint32_t *blob,
    uint32_t *projective,
    uint32_t *vars,
    uint32_t *error
) {
    if (*error != 0) return;
    size_t flat_row = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t total_rows = num_instructions * EC_MUL_COMPUTE_ROWS;
    if (flat_row >= total_rows || projection[flat_row / EC_MUL_COMPUTE_ROWS].is_setup != 0)
        return;
    FieldExprProg s;
    load_prog(blob, s);
    ec_mul_projective_materialize_row<K>(s, projective, vars, total_rows, flat_row);
}

template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __global__ void ec_mul_projective_setup_vars(
    const EcMulTraceInput<BLOCKS> *projection,
    size_t num_instructions,
    const uint32_t *blob,
    uint32_t *vars,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    uint32_t *error
) {
    if (*error != 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    FieldExprProg s;
    load_prog(blob, s);
    if (aux_words > scratch_words) {
        preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return;
    }
    uint32_t *aux = scratch;
    uint8_t in_limbs[EC_MUL_EXPR_NUM_INPUTS * NUM_LIMBS];
    size_t total_rows = num_instructions * EC_MUL_COMPUTE_ROWS;
    for (size_t instruction = 0; instruction < num_instructions; instruction++) {
        if (projection[instruction].is_setup == 0) continue;
        ec_mul_setup_inputs(in_limbs, s);
        if (!field_expr_eval_values<K>(
                s, in_limbs, nullptr, FieldExprRowMode{0, true, false}, aux, error
            )) return;
        size_t first_row = instruction * EC_MUL_COMPUTE_ROWS;
        for (uint32_t word = 0; word < s.num_vars * K; word++) {
            for (size_t row = 0; row < EC_MUL_COMPUTE_ROWS; row++) {
                vars[word * total_rows + first_row + row] = aux[word];
            }
        }
    }
}

template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static int launch_ec_mul_projective_vars(
    const void *projection,
    size_t num_instructions,
    const uint32_t *blob,
    uint32_t *vars,
    size_t vars_words,
    uint32_t *projective,
    size_t projective_words,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    uint32_t *error,
    cudaStream_t stream
) {
    if (projection == nullptr || blob == nullptr || vars == nullptr || projective == nullptr ||
        scratch == nullptr || error == nullptr || num_instructions == 0 ||
        vars_words == 0 ||
        projective_words < num_instructions * EC_MUL_PROJECTIVE_INSTRUCTION_WORDS<K> ||
        scratch_words < aux_words)
        return cudaErrorInvalidValue;
    auto *inputs = static_cast<const EcMulTraceInput<BLOCKS> *>(projection);
    auto [instruction_grid, instruction_block] = kernel_launch_params(num_instructions, 64);
    ec_mul_projective_prepare_pass<K, BLOCKS><<<instruction_grid, instruction_block, 0, stream>>>(
        inputs, num_instructions, blob, projective, error
    );
    if (int result = CHECK_KERNEL(); result != 0) return result;
    ec_mul_projective_batch_invert_pass<K, NUM_LIMBS, BLOCKS>
        <<<instruction_grid, instruction_block, 0, stream>>>(
            inputs, num_instructions, blob, projective, error
        );
    if (int result = CHECK_KERNEL(); result != 0) return result;
    auto [row_grid, row_block] =
        kernel_launch_params(num_instructions * EC_MUL_COMPUTE_ROWS, 128);
    ec_mul_projective_materialize_pass<K, NUM_LIMBS, BLOCKS>
        <<<row_grid, row_block, 0, stream>>>(
            inputs, num_instructions, blob, projective, vars, error
        );
    if (int result = CHECK_KERNEL(); result != 0) return result;
    ec_mul_projective_setup_vars<K, NUM_LIMBS, BLOCKS><<<1, 1, 0, stream>>>(
            inputs, num_instructions, blob, vars, scratch, scratch_words, aux_words, error
        );
    return CHECK_KERNEL();
}

extern "C" int _ec_mul_projective_generate_vars(
    size_t num_limbs,
    size_t blocks,
    const void *projection,
    size_t num_instructions,
    const uint32_t *blob,
    uint32_t *vars,
    size_t vars_words,
    uint32_t *projective,
    size_t projective_words,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    uint32_t *error,
    cudaStream_t stream
) {
    if (num_limbs == 32 && blocks == 8) {
        return launch_ec_mul_projective_vars<8, 32, 8>(
            projection, num_instructions, blob, vars, vars_words, projective, projective_words,
            scratch, scratch_words, aux_words, error, stream
        );
    }
    if (num_limbs == 48 && blocks == 12) {
        return launch_ec_mul_projective_vars<12, 48, 12>(
            projection, num_instructions, blob, vars, vars_words, projective, projective_words,
            scratch, scratch_words, aux_words, error, stream
        );
    }
    return cudaErrorInvalidValue;
}

// Checks the shape the host derived against the blob, before any row is written.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __global__ void ec_mul_validate(
    size_t height,
    size_t width,
    size_t num_instructions,
    size_t vars_words,
    const uint32_t *blob,
    size_t blob_words,
    size_t aux_words,
    uint32_t *error
) {
    FieldExprProg s;
    if (!validate_and_load_prog(blob, blob_words, s) || aux_words != s.aux_words ||
        !ec_mul_validate_trace_shape<K, NUM_LIMBS, BLOCKS>(
            s, width, height, num_instructions, vars_words
        )) {
        preflight_set_error(error, EC_MUL_BAD_PROGRAM);
    }
}

// Builds the inactive expression witness once, for every digest and padding row to share.
template <uint32_t K, size_t NUM_LIMBS>
static __global__ void ec_mul_dummy_expr(
    Fp *dummy,
    const uint32_t *blob,
    uint32_t *discarded_counts,
    size_t range_bins,
    uint32_t *scratch,
    uint32_t *error
) {
    if (*error != 0) return;
    FieldExprProg s;
    load_prog(blob, s);
    uint8_t in_limbs[EC_MUL_EXPR_NUM_INPUTS * NUM_LIMBS];
    VariableRangeChecker discarded(discarded_counts, range_bins);
    ec_mul_build_dummy_expr<K>(s, RowSlice(dummy, 1), discarded, scratch, in_limbs, error);
}

// One thread per trace row, grid-stride so the scratch a launch needs is bounded by its thread
// count rather than by the trace height.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __global__ void ec_mul_fill(
    Fp *trace,
    size_t height,
    size_t num_instructions,
    const EcMulTraceInput<BLOCKS> *projection,
    const uint32_t *blob,
    const uint32_t *vars,
    bool vars_transposed,
    const Fp *dummy_expr,
    uint32_t *range_counts,
    size_t range_bins,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    if (*error != 0) return;
    FieldExprProg s;
    load_prog(blob, s);

    const size_t threads = gridDim.x * static_cast<size_t>(blockDim.x);
    if (threads == 0 || threads * aux_words > scratch_words) {
        preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return;
    }
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    VariableRangeChecker range_checker(range_counts, range_bins);
    uint32_t *aux = scratch + tid * aux_words;
    uint8_t in_limbs[EC_MUL_EXPR_NUM_INPUTS * NUM_LIMBS];
    uint8_t accumulator[2 * NUM_LIMBS];
    const size_t used_rows = num_instructions * EC_MUL_TOTAL_ROWS;

    for (size_t row_index = tid; row_index < height; row_index += threads) {
        if (!ec_mul_fill_row<K, NUM_LIMBS, BLOCKS>(
                s,
                RowSlice(trace + row_index, height),
                row_index,
                used_rows,
                projection,
                vars,
                vars_transposed,
                dummy_expr,
                range_checker,
                timestamp_max_bits,
                pointer_max_bits,
                aux,
                in_limbs,
                accumulator,
                error
            )) {
            return;
        }
    }
}

template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static int launch_ec_mul_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    const void *projection,
    size_t num_instructions,
    const uint32_t *blob,
    size_t blob_words,
    const uint32_t *vars,
    size_t vars_words,
    bool vars_transposed,
    Fp *dummy_expr,
    uint32_t *range_counts,
    size_t range_bins,
    uint32_t *discarded_counts,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    size_t fill_grid_blocks,
    size_t fill_block_threads,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (trace == nullptr || projection == nullptr || blob == nullptr || vars == nullptr ||
        dummy_expr == nullptr || range_counts == nullptr || discarded_counts == nullptr ||
        scratch == nullptr || error == nullptr || height == 0 || num_instructions == 0 ||
        aux_words == 0 || fill_grid_blocks == 0 || fill_block_threads == 0 ||
        fill_grid_blocks > UINT32_MAX || fill_block_threads > 1024) {
        return cudaErrorInvalidValue;
    }
    if (num_instructions > height / EC_MUL_TOTAL_ROWS ||
        fill_grid_blocks * fill_block_threads * aux_words > scratch_words) {
        return cudaErrorInvalidValue;
    }
    auto *inputs = static_cast<const EcMulTraceInput<BLOCKS> *>(projection);

    ec_mul_validate<K, NUM_LIMBS, BLOCKS><<<1, 1, 0, stream>>>(
        height, width, num_instructions, vars_words, blob, blob_words, aux_words, error
    );
    if (int result = CHECK_KERNEL(); result != 0) return result;

    ec_mul_dummy_expr<K, NUM_LIMBS><<<1, 1, 0, stream>>>(
        dummy_expr, blob, discarded_counts, range_bins, scratch, error
    );
    if (int result = CHECK_KERNEL(); result != 0) return result;

    ec_mul_fill<K, NUM_LIMBS, BLOCKS>
        <<<static_cast<uint32_t>(fill_grid_blocks),
           static_cast<uint32_t>(fill_block_threads),
           0,
           stream>>>(
            trace,
            height,
            num_instructions,
            inputs,
            blob,
            vars,
            vars_transposed,
            dummy_expr,
            range_counts,
            range_bins,
            scratch,
            scratch_words,
            aux_words,
            pointer_max_bits,
            timestamp_max_bits,
            error
        );
    return CHECK_KERNEL();
}

extern "C" int _ec_mul_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    size_t num_limbs,
    size_t blocks,
    const void *projection,
    size_t num_instructions,
    const uint32_t *blob,
    size_t blob_words,
    const uint32_t *vars,
    size_t vars_words,
    bool vars_transposed,
    Fp *dummy_expr,
    uint32_t *range_counts,
    size_t range_bins,
    uint32_t *discarded_counts,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    size_t fill_grid_blocks,
    size_t fill_block_threads,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (num_limbs == 32 && blocks == 8) {
        return launch_ec_mul_tracegen<8, 32, 8>(
            trace, height, width, projection, num_instructions, blob, blob_words, vars, vars_words,
            vars_transposed, dummy_expr, range_counts, range_bins, discarded_counts, scratch, scratch_words,
            aux_words, fill_grid_blocks, fill_block_threads, pointer_max_bits, timestamp_max_bits,
            error, stream
        );
    }
    if (num_limbs == 48 && blocks == 12) {
        return launch_ec_mul_tracegen<12, 48, 12>(
            trace, height, width, projection, num_instructions, blob, blob_words, vars, vars_words,
            vars_transposed, dummy_expr, range_counts, range_bins, discarded_counts, scratch, scratch_words,
            aux_words, fill_grid_blocks, fill_block_threads, pointer_max_bits, timestamp_max_bits,
            error, stream
        );
    }
    return cudaErrorInvalidValue;
}
