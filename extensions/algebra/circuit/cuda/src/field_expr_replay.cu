// Generic GPU tracegen for mod-builder FieldExpr chips whose trace is one row per instruction.
// One thread per row, grid-stride. The expression interpreter itself lives in
// `algebra/field_expr_core.cuh`; this file supplies the vec-heap adapter columns around it.
#include "algebra/field_expr_core.cuh"
#include "algebra/vec_heap_replay.cuh"

template <size_t NUM_READS, size_t BLOCKS>
static __global__ void validate_field_expr_replay(
    size_t height,
    size_t width,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    size_t aux_words,
    uint32_t *error
) {
    constexpr uint32_t K = BLOCKS <= 6 ? 2 * BLOCKS : BLOCKS;
    FieldExprProg s;
    if (!validate_and_load_prog(blob, blob_words, s)) {
        preflight_set_error(error, FIELD_EXPR_BAD_BLOB);
        return;
    }
    constexpr size_t ADAPTER_WIDTH =
        sizeof(VecHeapAdapterCols<uint8_t, NUM_READS, BLOCKS, BLOCKS>);
    constexpr size_t INPUT_BYTES = NUM_READS * BLOCKS * MEMORY_BLOCK_BYTES;
    constexpr size_t OUTPUT_BYTES = BLOCKS * MEMORY_BLOCK_BYTES;
    if (s.k != K || width != ADAPTER_WIDTH + s.width ||
        static_cast<uint64_t>(s.num_input) * s.num_limbs != INPUT_BYTES ||
        static_cast<uint64_t>(s.n_outputs) * s.num_limbs != OUTPUT_BYTES ||
        projection_len > height || aux_words != s.aux_words) {
        preflight_set_error(error, FIELD_EXPR_BAD_TRACE_SHAPE);
    }
}

template <size_t NUM_READS, size_t BLOCKS>
static __global__ void field_expr_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    const VecHeapTraceInput<NUM_READS, BLOCKS> *projection,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    uint32_t *range_delta,
    size_t range_bins,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    constexpr uint32_t K = BLOCKS <= 6 ? 2 * BLOCKS : BLOCKS;
    constexpr size_t ADAPTER_WIDTH =
        sizeof(VecHeapAdapterCols<uint8_t, NUM_READS, BLOCKS, BLOCKS>);
    __shared__ FieldExprProg shared_program;
    if (threadIdx.x == 0) load_prog(blob, shared_program);
    __syncthreads();
    if (*error != 0) return;
    const FieldExprProg &s = shared_program;
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t nthreads = gridDim.x * (size_t)blockDim.x;
    if (nthreads == 0 ||
        static_cast<uint64_t>(nthreads) * aux_words > scratch_words) {
        preflight_set_error(error, FIELD_EXPR_BAD_TRACE_SHAPE);
        return;
    }
    VariableRangeChecker range_checker(range_delta, range_bins);
    uint32_t *thread_scratch = scratch + tid * aux_words;

    for (size_t row_index = tid; row_index < height; row_index += nthreads) {
        RowSlice row(trace + row_index, height);
        row.fill_zero(0, width);
        if (row_index < projection_len) {
            auto const &input = projection[row_index];
            const uint8_t *input_limbs =
                reinterpret_cast<const uint8_t *>(&input.heap_reads[0][0][0]);
            const uint8_t *logged_output =
                reinterpret_cast<const uint8_t *>(&input.writes[0][0]);
            if (!field_expr_fill_core_row<K>(
                    s,
                    row.slice_from(ADAPTER_WIDTH),
                    input_limbs,
                    logged_output,
                    input.local_opcode,
                    range_checker,
                    thread_scratch,
                    false,
                    error
                )) {
                return;
            }
            fill_vec_heap_adapter_from_projection(
                row, input, range_checker, pointer_max_bits, timestamp_max_bits
            );
        } else {
            if (s.should_finalize &&
                !field_expr_fill_core_row<K>(
                    s,
                    row.slice_from(ADAPTER_WIDTH),
                    nullptr,
                    nullptr,
                    UINT32_MAX,
                    range_checker,
                    thread_scratch,
                    true,
                    error
                )) {
                return;
            }
        }
    }
}

template <size_t NUM_READS, size_t BLOCKS>
static int field_expr_kernel_config(
    size_t *max_grid_blocks,
    size_t *block_threads,
    size_t *local_bytes_per_thread
) {
    if (max_grid_blocks == nullptr || block_threads == nullptr ||
        local_bytes_per_thread == nullptr) {
        return cudaErrorInvalidValue;
    }
    static constexpr int THREADS = 128;
    // These properties depend only on the selected device and kernel variant. The host caches
    // them with the chip and derives height/scratch-limited launch dimensions per trace.
    int blocks_per_multiprocessor;
    cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor,
        field_expr_replay_tracegen<NUM_READS, BLOCKS>,
        THREADS,
        0
    );
    if (result != cudaSuccess) return result;
    int device;
    result = cudaGetDevice(&device);
    if (result != cudaSuccess) return result;
    int multiprocessors;
    result =
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (result != cudaSuccess) return result;
    cudaFuncAttributes attributes;
    result = cudaFuncGetAttributes(
        &attributes, field_expr_replay_tracegen<NUM_READS, BLOCKS>
    );
    if (result != cudaSuccess) return result;
    size_t resident_blocks =
        static_cast<size_t>(blocks_per_multiprocessor) * multiprocessors;
    if (resident_blocks == 0) return cudaErrorInvalidValue;
    *max_grid_blocks = resident_blocks;
    *block_threads = THREADS;
    *local_bytes_per_thread = attributes.localSizeBytes;
    return cudaSuccess;
}

extern "C" int _field_expr_replay_kernel_config(
    size_t num_reads,
    size_t blocks,
    size_t *max_grid_blocks,
    size_t *block_threads,
    size_t *local_bytes_per_thread
) {
    if (num_reads == 2 && blocks == 4)
        return field_expr_kernel_config<2, 4>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 2 && blocks == 6)
        return field_expr_kernel_config<2, 6>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 2 && blocks == 8)
        return field_expr_kernel_config<2, 8>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 2 && blocks == 12)
        return field_expr_kernel_config<2, 12>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 1 && blocks == 8)
        return field_expr_kernel_config<1, 8>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 1 && blocks == 12)
        return field_expr_kernel_config<1, 12>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    return cudaErrorInvalidValue;
}

template <size_t NUM_READS, size_t BLOCKS>
static int launch_field_expr_replay(
    Fp *trace,
    size_t height,
    size_t width,
    const void *projection,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    uint32_t *range_delta,
    size_t range_bins,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    size_t grid_blocks,
    size_t block_threads,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (trace == nullptr || projection == nullptr || blob == nullptr ||
        range_delta == nullptr || scratch == nullptr || error == nullptr ||
        grid_blocks == 0 || block_threads == 0 || grid_blocks > UINT32_MAX ||
        block_threads > 1024) {
        return cudaErrorInvalidValue;
    }
    validate_field_expr_replay<NUM_READS, BLOCKS><<<1, 1, 0, stream>>>(
        height, width, projection_len, blob, blob_words, aux_words, error
    );
    if (int result = CHECK_KERNEL(); result != 0) return result;
    field_expr_replay_tracegen<NUM_READS, BLOCKS>
        <<<static_cast<uint32_t>(grid_blocks),
           static_cast<uint32_t>(block_threads),
           0,
           stream>>>(
            trace,
            height,
            width,
            static_cast<const VecHeapTraceInput<NUM_READS, BLOCKS> *>(projection),
            projection_len,
            blob,
            blob_words,
            range_delta,
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

extern "C" int _field_expr_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    size_t num_reads,
    size_t blocks,
    const void *projection,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    uint32_t *range_delta,
    size_t range_bins,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    size_t grid_blocks,
    size_t block_threads,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (num_reads == 2 && blocks == 4)
        return launch_field_expr_replay<2, 4>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 2 && blocks == 6)
        return launch_field_expr_replay<2, 6>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 2 && blocks == 8)
        return launch_field_expr_replay<2, 8>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 2 && blocks == 12)
        return launch_field_expr_replay<2, 12>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 1 && blocks == 8)
        return launch_field_expr_replay<1, 8>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 1 && blocks == 12)
        return launch_field_expr_replay<1, 12>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    return cudaErrorInvalidValue;
}
