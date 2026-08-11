// Trace generation for the `EC_MUL` chip. The per-row and per-instruction bodies are in
// `algebra/ec_mul_tracegen.cuh`; this file supplies the kernels and the launch policy.
//
// Like the `EC_MUL` projection gather, this lives in the algebra crate because that is the crate
// compiling CUDA for the extension stack; the ECC circuit crate has no CUDA build of its own.
#include "algebra/ec_mul_tracegen.cuh"
#include "launcher.cuh"

// Checks the shape the host derived against the blob, before any row is written.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __global__ void ec_mul_validate(
    size_t height,
    size_t width,
    size_t num_instructions,
    size_t affine_bytes,
    size_t ladder_words,
    const uint32_t *blob,
    size_t blob_words,
    size_t aux_words,
    uint32_t *error
) {
    FieldExprProg s;
    if (!validate_and_load_prog(blob, blob_words, s) || aux_words != s.aux_words ||
        !ec_mul_validate_trace_shape<K, NUM_LIMBS, BLOCKS>(
            s, width, height, num_instructions, affine_bytes, ladder_words
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

// One thread per instruction: advances the ladder, storing each row's saved variables.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __global__ void ec_mul_eval(
    const EcMulTraceInput<BLOCKS> *projection,
    size_t num_instructions,
    const uint32_t *blob,
    uint8_t *affine,
    uint32_t *ladder,
    uint32_t *error
) {
    if (*error != 0) return;
    size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (index >= num_instructions) return;

    FieldExprProg s;
    load_prog(blob, s);
    // Both the Jacobian scratch and the Montgomery temporaries come from the per-instruction slice,
    // rather than a kernel-local array large enough to hurt occupancy.
    uint8_t *row_affine = affine + index * EC_MUL_COMPUTE_ROWS * 2 * NUM_LIMBS;
    uint32_t *row_ladder = ladder + index * EC_MUL_LADDER_SLICE_WORDS<K>;

    ec_mul_eval_instruction<K, BLOCKS>(
        s, projection[index], row_affine, row_ladder,
        row_ladder + EC_MUL_COMPUTE_ROWS * 4 * K, error
    );
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
    const uint8_t *affine,
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
    const size_t used_rows = num_instructions * EC_MUL_TOTAL_ROWS;

    for (size_t row_index = tid; row_index < height; row_index += threads) {
        if (!ec_mul_fill_row<K, NUM_LIMBS, BLOCKS>(
                s,
                RowSlice(trace + row_index, height),
                row_index,
                used_rows,
                projection,
                affine,
                dummy_expr,
                range_checker,
                timestamp_max_bits,
                pointer_max_bits,
                aux,
                in_limbs,
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
    uint8_t *affine,
    size_t affine_bytes,
    uint32_t *ladder,
    size_t ladder_words,
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
    if (trace == nullptr || projection == nullptr || blob == nullptr || affine == nullptr ||
        ladder == nullptr || dummy_expr == nullptr || range_counts == nullptr ||
        discarded_counts == nullptr || scratch == nullptr || error == nullptr || height == 0 ||
        num_instructions == 0 ||
        aux_words == 0 || fill_grid_blocks == 0 || fill_block_threads == 0 ||
        fill_grid_blocks > UINT32_MAX || fill_block_threads > 1024) {
        return cudaErrorInvalidValue;
    }
    const size_t needed_affine = num_instructions * EC_MUL_COMPUTE_ROWS * 2 * NUM_LIMBS;
    const size_t needed_ladder = num_instructions * EC_MUL_LADDER_SLICE_WORDS<K>;
    if (num_instructions > height / EC_MUL_TOTAL_ROWS ||
        fill_grid_blocks * fill_block_threads * aux_words > scratch_words ||
        affine_bytes < needed_affine || ladder_words < needed_ladder) {
        return cudaErrorInvalidValue;
    }
    auto *inputs = static_cast<const EcMulTraceInput<BLOCKS> *>(projection);

    ec_mul_validate<K, NUM_LIMBS, BLOCKS><<<1, 1, 0, stream>>>(
        height, width, num_instructions, affine_bytes, ladder_words, blob, blob_words, aux_words,
        error
    );
    if (int result = CHECK_KERNEL(); result != 0) return result;

    ec_mul_dummy_expr<K, NUM_LIMBS><<<1, 1, 0, stream>>>(
        dummy_expr, blob, discarded_counts, range_bins, scratch, error
    );
    if (int result = CHECK_KERNEL(); result != 0) return result;

    auto [eval_grid, eval_block] = kernel_launch_params(num_instructions, 64);
    ec_mul_eval<K, NUM_LIMBS, BLOCKS><<<eval_grid, eval_block, 0, stream>>>(
        inputs, num_instructions, blob, affine, ladder, error
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
            affine,
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
    uint8_t *affine,
    size_t affine_bytes,
    uint32_t *ladder,
    size_t ladder_words,
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
            trace, height, width, projection, num_instructions, blob, blob_words, affine,
            affine_bytes, ladder, ladder_words, dummy_expr, range_counts, range_bins,
            discarded_counts, scratch, scratch_words, aux_words, fill_grid_blocks,
            fill_block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    }
    if (num_limbs == 48 && blocks == 12) {
        return launch_ec_mul_tracegen<12, 48, 12>(
            trace, height, width, projection, num_instructions, blob, blob_words, affine,
            affine_bytes, ladder, ladder_words, dummy_expr, range_counts, range_bins,
            discarded_counts, scratch, scratch_words, aux_words, fill_grid_blocks,
            fill_block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    }
    return cudaErrorInvalidValue;
}
