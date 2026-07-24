#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv-adapters/vec_heap.cuh"
#include "riscv-adapters/vec_heap_replay.cuh"

#include <cstddef>
#include <cstdint>

static constexpr uint32_t MODULAR_ADDSUB_REPLAY_ERROR = 0x4d020001;
static constexpr uint32_t BABYBEAR_MODULUS = 0x78000001;

static __device__ Fp signed_fp(int32_t value) {
    return Fp(
        value >= 0 ? static_cast<uint32_t>(value)
                   : BABYBEAR_MODULUS - static_cast<uint32_t>(-value)
    );
}

template <size_t BYTES>
static __device__ int compare_bytes(uint8_t const (&a)[BYTES], uint8_t const *b) {
    for (int i = static_cast<int>(BYTES) - 1; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

template <size_t BYTES>
static __device__ uint32_t subtract_bytes(
    uint8_t (&out)[BYTES], uint8_t const (&a)[BYTES], uint8_t const *b
) {
    uint32_t borrow = 0;
    for (size_t i = 0; i < BYTES; i++) {
        uint32_t rhs = static_cast<uint32_t>(b[i]) + borrow;
        out[i] = static_cast<uint8_t>(static_cast<uint32_t>(a[i]) - rhs);
        borrow = static_cast<uint32_t>(a[i]) < rhs;
    }
    return borrow;
}

template <size_t BYTES>
static __device__ bool reduce_bytes(
    uint8_t const (&input)[BYTES],
    uint8_t const *modulus,
    uint8_t (&reduced)[BYTES],
    uint32_t &quotient
) {
    for (size_t i = 0; i < BYTES; i++) reduced[i] = input[i];
    quotient = 0;
    while (compare_bytes(reduced, modulus) >= 0) {
        if (quotient == UINT8_MAX) return false;
        uint8_t next[BYTES];
        if (subtract_bytes(next, reduced, modulus) != 0) return false;
        for (size_t i = 0; i < BYTES; i++) reduced[i] = next[i];
        quotient++;
    }
    return true;
}

template <size_t BYTES>
static __device__ bool add_mod(
    uint8_t const (&a)[BYTES],
    uint8_t const (&b)[BYTES],
    uint8_t const *modulus,
    uint8_t (&out)[BYTES],
    uint32_t &quotient
) {
    uint32_t carry = 0;
    for (size_t i = 0; i < BYTES; i++) {
        uint32_t value = static_cast<uint32_t>(a[i]) + b[i] + carry;
        out[i] = static_cast<uint8_t>(value);
        carry = value >> 8;
    }
    quotient = carry || compare_bytes(out, modulus) >= 0;
    if (quotient) {
        uint8_t reduced[BYTES];
        subtract_bytes(reduced, out, modulus);
        for (size_t i = 0; i < BYTES; i++) out[i] = reduced[i];
    }
    return true;
}

template <size_t BYTES>
static __device__ void sub_mod(
    uint8_t const (&a)[BYTES],
    uint8_t const (&b)[BYTES],
    uint8_t const *modulus,
    uint8_t (&out)[BYTES],
    uint32_t &borrow
) {
    borrow = compare_bytes(a, b) < 0;
    if (!borrow) {
        subtract_bytes(out, a, b);
        return;
    }
    uint8_t difference[BYTES];
    subtract_bytes(difference, b, a);
    uint8_t modulus_array[BYTES];
    for (size_t i = 0; i < BYTES; i++) modulus_array[i] = modulus[i];
    subtract_bytes(out, modulus_array, difference);
}

template <size_t NUM_READS, size_t BLOCKS>
static __device__ void fill_adapter_from_projection(
    RowSlice row,
    VecHeapTraceInput<NUM_READS, BLOCKS> const &input,
    VariableRangeChecker range_checker,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    Rv64VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS> record = {};
    record.from_pc = input.from_pc;
    record.from_timestamp = input.from_timestamp;
    record.rd_ptr = input.rd_ptr;
    record.rd_val = input.rd_val;
    record.rd_read_aux.prev_timestamp = input.rd_prev_timestamp;
    for (size_t read = 0; read < NUM_READS; read++) {
        record.rs_ptrs[read] = input.rs_ptrs[read];
        record.rs_vals[read] = input.rs_vals[read];
        record.rs_read_aux[read].prev_timestamp = input.rs_prev_timestamps[read];
        for (size_t block = 0; block < BLOCKS; block++) {
            record.reads_aux[read][block].prev_timestamp =
                input.heap_prev_timestamps[read][block];
        }
    }
    for (size_t block = 0; block < BLOCKS; block++) {
        record.writes_aux[block].prev_timestamp = input.write_prev_timestamps[block];
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            uint16_t packed = input.write_predecessors[block][limb];
            record.writes_aux[block].prev_data[2 * limb] = static_cast<uint8_t>(packed);
            record.writes_aux[block].prev_data[2 * limb + 1] =
                static_cast<uint8_t>(packed >> 8);
        }
    }
    Rv64VecHeapAdapter<NUM_READS, BLOCKS, BLOCKS> adapter(
        pointer_max_bits, range_checker, timestamp_max_bits
    );
    adapter.fill_trace_row(row, record);
}

template <size_t BLOCKS>
__global__ void modular_addsub_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<VecHeapTraceInput<2, BLOCKS>> projection,
    uint8_t const *modulus,
    uint32_t add_local_opcode,
    uint32_t sub_local_opcode,
    uint32_t setup_local_opcode,
    uint32_t *range_checker_counts,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    constexpr size_t BYTES = BLOCKS * MEMORY_BLOCK_BYTES;
    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv64VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS>);
    constexpr size_t CORE_WIDTH = 4 * BYTES + 4;
    size_t row_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row_index >= height) return;
    RowSlice row(d_trace + row_index, height);
    row.fill_zero(0, width);
    if (row_index >= projection.len()) return;
    if (width != ADAPTER_WIDTH + CORE_WIDTH) {
        preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
        return;
    }

    auto const &input = projection[row_index];
    bool is_add = input.local_opcode == add_local_opcode;
    bool is_sub = input.local_opcode == sub_local_opcode;
    bool is_setup = input.local_opcode == setup_local_opcode;
    if ((!is_add && !is_sub && !is_setup) || modulus[0] == 0) {
        preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
        return;
    }

    uint8_t x[BYTES];
    uint8_t y[BYTES];
    uint8_t logged[BYTES];
    for (size_t block = 0; block < BLOCKS; block++) {
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            size_t byte = block * MEMORY_BLOCK_BYTES + 2 * limb;
            uint16_t x_packed = input.heap_reads[0][block][limb];
            uint16_t y_packed = input.heap_reads[1][block][limb];
            uint16_t z_packed = input.writes[block][limb];
            x[byte] = static_cast<uint8_t>(x_packed);
            x[byte + 1] = static_cast<uint8_t>(x_packed >> 8);
            y[byte] = static_cast<uint8_t>(y_packed);
            y[byte + 1] = static_cast<uint8_t>(y_packed >> 8);
            logged[byte] = static_cast<uint8_t>(z_packed);
            logged[byte + 1] = static_cast<uint8_t>(z_packed >> 8);
        }
    }

    uint8_t x_reduced[BYTES];
    uint8_t y_reduced[BYTES];
    uint8_t expected[BYTES];
    uint32_t x_quotient;
    uint32_t y_quotient;
    if (!reduce_bytes(x, modulus, x_reduced, x_quotient) ||
        !reduce_bytes(y, modulus, y_reduced, y_quotient)) {
        preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
        return;
    }

    int32_t quotient;
    if (is_setup) {
        for (size_t i = 0; i < BYTES; i++) {
            if (x[i] != modulus[i]) {
                preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
                return;
            }
            expected[i] = 0;
        }
        quotient = 1;
    } else if (is_add) {
        uint32_t reduced_quotient;
        add_mod(x_reduced, y_reduced, modulus, expected, reduced_quotient);
        uint32_t q = x_quotient + y_quotient + reduced_quotient;
        if (q > UINT8_MAX) {
            preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
            return;
        }
        quotient = static_cast<int32_t>(q);
    } else {
        uint32_t borrow;
        sub_mod(x_reduced, y_reduced, modulus, expected, borrow);
        quotient = static_cast<int32_t>(x_quotient) -
                   static_cast<int32_t>(y_quotient) - static_cast<int32_t>(borrow);
        if (quotient < -static_cast<int32_t>(UINT8_MAX) - 1 ||
            quotient > static_cast<int32_t>(UINT8_MAX)) {
            preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
            return;
        }
    }
    for (size_t i = 0; i < BYTES; i++) {
        if (logged[i] != expected[i]) {
            preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
            return;
        }
    }

    int32_t carries[BYTES];
    int32_t carry = 0;
    for (size_t i = 0; i < BYTES; i++) {
        int32_t expression = static_cast<int32_t>(x[i]) - logged[i];
        if (is_add) expression += y[i];
        if (is_sub) expression -= y[i];
        expression -= quotient * static_cast<int32_t>(modulus[i]);
        int32_t value = expression + carry;
        carry = value >= 0 ? value >> 8 : -((-value + 255) >> 8);
        carries[i] = carry;
    }
    if (carry != 0) {
        preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
        return;
    }

    VariableRangeChecker range_checker(range_checker_counts, range_checker_bins);
    fill_adapter_from_projection(
        row, input, range_checker, pointer_max_bits, timestamp_max_bits
    );
    RowSlice core = row.slice_from(ADAPTER_WIDTH);
    size_t column = 0;
    core[column++] = Fp::one();
    for (size_t i = 0; i < BYTES; i++) core[column++] = Fp(x[i]);
    for (size_t i = 0; i < BYTES; i++) core[column++] = Fp(y[i]);
    for (size_t i = 0; i < BYTES; i++) {
        core[column++] = Fp(logged[i]);
        range_checker.add_count(logged[i], 8);
    }
    core[column++] = signed_fp(quotient);
    range_checker.add_count(static_cast<uint32_t>(quotient + 256), 9);
    for (size_t i = 0; i < BYTES; i++) {
        core[column++] = signed_fp(carries[i]);
        range_checker.add_count(static_cast<uint32_t>(carries[i] + 512), 10);
    }
    core[column++] = Fp(static_cast<uint32_t>(is_add));
    core[column++] = Fp(static_cast<uint32_t>(is_sub));
    if (column != CORE_WIDTH) preflight_set_error(error, MODULAR_ADDSUB_REPLAY_ERROR);
}

template <size_t BLOCKS>
static int launch_modular_addsub_replay(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<VecHeapTraceInput<2, BLOCKS>> projection,
    uint8_t const *modulus,
    uint32_t add_local_opcode,
    uint32_t sub_local_opcode,
    uint32_t setup_local_opcode,
    uint32_t *range_checker_counts,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (projection.len() == 0 || height == 0) return 0;
    auto [grid, block] = kernel_launch_params(height, 256);
    modular_addsub_replay_tracegen<BLOCKS><<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        projection,
        modulus,
        add_local_opcode,
        sub_local_opcode,
        setup_local_opcode,
        range_checker_counts,
        range_checker_bins,
        pointer_max_bits,
        timestamp_max_bits,
        error
    );
    return CHECK_KERNEL();
}

extern "C" int _modular_addsub_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    size_t blocks,
    void const *projection,
    size_t projection_len,
    uint8_t const *modulus,
    uint32_t add_local_opcode,
    uint32_t sub_local_opcode,
    uint32_t setup_local_opcode,
    uint32_t *range_checker_counts,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (blocks == 4) {
        return launch_modular_addsub_replay<4>(
            d_trace,
            height,
            width,
            DeviceBufferConstView<VecHeapTraceInput<2, 4>>{
                reinterpret_cast<VecHeapTraceInput<2, 4> const *>(projection),
                projection_len * sizeof(VecHeapTraceInput<2, 4>)
            },
            modulus,
            add_local_opcode,
            sub_local_opcode,
            setup_local_opcode,
            range_checker_counts,
            range_checker_bins,
            pointer_max_bits,
            timestamp_max_bits,
            error,
            stream
        );
    }
    if (blocks == 6) {
        return launch_modular_addsub_replay<6>(
            d_trace,
            height,
            width,
            DeviceBufferConstView<VecHeapTraceInput<2, 6>>{
                reinterpret_cast<VecHeapTraceInput<2, 6> const *>(projection),
                projection_len * sizeof(VecHeapTraceInput<2, 6>)
            },
            modulus,
            add_local_opcode,
            sub_local_opcode,
            setup_local_opcode,
            range_checker_counts,
            range_checker_bins,
            pointer_max_bits,
            timestamp_max_bits,
            error,
            stream
        );
    }
    return 1;
}
