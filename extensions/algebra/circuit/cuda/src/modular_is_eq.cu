// GPU tracegen for ModularIsEqual chips (IsEqualMod core + Rv64IsEqualModU16 adapter).
// One thread per row; transliterates ModularIsEqualFiller::fill_trace_row and
// Rv32IsEqualModAdapterFiller::fill_trace_row.
#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32-adapters/vec_heap.cuh" // shared adapter/aux helpers

#include <cstdint>

// ---- column structs (mirror the Rust #[repr(C)] AlignedBorrow layouts) ----

template <typename T, size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv32IsEqualModAdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rs_val[NUM_READS][RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> heap_read_aux[NUM_READS][BLOCKS_PER_READ];

    T rd_ptr;
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> writes_aux;
};

template <typename T, size_t READ_LIMBS> struct ModularIsEqualCoreCols {
    T is_valid;
    T is_setup;
    T b[READ_LIMBS];
    T c[READ_LIMBS];
    T cmp_result;
    T eq_marker[READ_LIMBS];
    T lt_marker[READ_LIMBS];
    T b_lt_diff;
    T c_lt_diff;
    T c_lt_mark;
};

// ---- record structs (mirror the Rust #[repr(C)] records) ----

template <size_t NUM_READS, size_t BLOCKS_PER_READ> struct Rv32IsEqualModAdapterRecord {
    uint32_t from_pc;
    uint32_t timestamp;

    uint32_t rs_ptr[NUM_READS];
    uint32_t rs_val[NUM_READS];
    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord heap_read_aux[NUM_READS][BLOCKS_PER_READ];

    uint32_t rd_ptr;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> writes_aux;
};

template <size_t READ_LIMBS> struct ModularIsEqualRecord {
    uint8_t is_setup; // bool
    uint8_t b[READ_LIMBS];
    uint8_t c[READ_LIMBS];
};

// run_unsigned_less_than: returns diff index (or READ_LIMBS when equal); lt result via out param.
template <size_t READ_LIMBS>
__device__ inline size_t
unsigned_less_than(const uint8_t (&x)[READ_LIMBS], const uint8_t *y, bool &lt) {
    for (int i = READ_LIMBS - 1; i >= 0; i--) {
        if (x[i] != y[i]) {
            lt = x[i] < y[i];
            return (size_t)i;
        }
    }
    lt = false;
    return READ_LIMBS;
}

template <size_t NUM_READS, size_t BLOCKS_PER_READ, size_t READ_LIMBS>
__global__ void modular_is_eq_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t rows_used,
    const uint8_t *__restrict__ d_records,
    size_t rec_stride,
    size_t rec_core_offset,
    const uint8_t *__restrict__ d_modulus_limbs, // READ_LIMBS u8 values
    uint32_t *__restrict__ d_range_checker,
    size_t rc_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits) {
    using AdapterColsU8 = Rv32IsEqualModAdapterCols<uint8_t, NUM_READS, BLOCKS_PER_READ>;
    using CoreColsU8 = ModularIsEqualCoreCols<uint8_t, READ_LIMBS>;
    constexpr size_t ADAPTER_WIDTH = sizeof(AdapterColsU8);
    constexpr size_t WIDTH = ADAPTER_WIDTH + sizeof(CoreColsU8);

    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t nthreads = gridDim.x * (size_t)blockDim.x;
    VariableRangeChecker rc(d_range_checker, rc_bins);
    BitwiseOperationLookup bitwise(d_bitwise_lookup, bitwise_num_bits);
    MemoryAuxColsFactory mem_helper(rc, timestamp_max_bits);

    __shared__ uint8_t s_modulus[READ_LIMBS];
    for (size_t i = threadIdx.x; i < READ_LIMBS; i += blockDim.x)
        s_modulus[i] = d_modulus_limbs[i];
    __syncthreads();

    for (size_t row = tid; row < height; row += nthreads) {
        RowSlice slice(d_trace + row, height);
        if (row >= rows_used) {
            slice.fill_zero(0, WIDTH);
            continue;
        }
        const uint8_t *rec_bytes = d_records + row * rec_stride;
        const auto &rec = *(const Rv32IsEqualModAdapterRecord<NUM_READS, BLOCKS_PER_READ> *)
                              rec_bytes;
        const auto &core =
            *(const ModularIsEqualRecord<READ_LIMBS> *)(rec_bytes + rec_core_offset);

        // ---- adapter columns (mirror Rv32IsEqualModAdapterFiller) ----
#define ACOL(field) (offsetof(AdapterColsU8, field))
        {
            // Mirror Rv32IsEqualModAdapterFiller: paired bitwise range request on the
            // most-significant register limbs, shifted by (32 - pointer_max_bits).
            const uint32_t msl_shift = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
            const uint32_t limb_shift =
                RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits;
            bitwise.add_range(
                (rec.rs_val[0] >> msl_shift) << limb_shift,
                NUM_READS > 1 ? (rec.rs_val[1] >> msl_shift) << limb_shift : 0);
        }
        uint32_t timestamp = rec.timestamp + (NUM_READS + NUM_READS * BLOCKS_PER_READ) + 1;

        // write aux (reverse timestamp order)
        timestamp--;
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++)
            slice[ACOL(writes_aux.prev_data) + i] = Fp((uint32_t)rec.writes_aux.prev_data[i]);
        mem_helper.fill(
            slice.slice_from(ACOL(writes_aux)), rec.writes_aux.prev_timestamp, timestamp);
        slice[ACOL(rd_ptr)] = Fp(rec.rd_ptr);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            for (int j = BLOCKS_PER_READ - 1; j >= 0; j--) {
                timestamp--;
                mem_helper.fill(
                    slice.slice_from(ACOL(heap_read_aux[i][j])),
                    rec.heap_read_aux[i][j].prev_timestamp,
                    timestamp);
            }
        }
        for (int i = NUM_READS - 1; i >= 0; i--) {
            timestamp--;
            mem_helper.fill(
                slice.slice_from(ACOL(rs_read_aux[i])),
                rec.rs_read_aux[i].prev_timestamp,
                timestamp);
        }
        for (size_t i = 0; i < NUM_READS; i++) {
            for (size_t j = 0; j < RV32_REGISTER_NUM_LIMBS; j++)
                slice[ACOL(rs_val[i]) + j] = Fp((rec.rs_val[i] >> (8 * j)) & 0xffu);
            slice[ACOL(rs_ptr[i])] = Fp(rec.rs_ptr[i]);
        }
        slice[ACOL(from_state.timestamp)] = Fp(rec.timestamp);
        slice[ACOL(from_state.pc)] = Fp(rec.from_pc);
#undef ACOL

        // ---- core columns (mirror ModularIsEqualFiller) ----
        RowSlice core_row = slice.slice_from(ADAPTER_WIDTH);
#define CCOL(field) (offsetof(CoreColsU8, field))
        bool b_lt, c_lt;
        const size_t b_diff_idx = unsigned_less_than<READ_LIMBS>(core.b, s_modulus, b_lt);
        const size_t c_diff_idx = unsigned_less_than<READ_LIMBS>(core.c, s_modulus, c_lt);

        const uint32_t c_lt_mark = (b_diff_idx == c_diff_idx) ? 1 : 2;
        core_row[CCOL(c_lt_mark)] = Fp(c_lt_mark);
        core_row[CCOL(c_lt_diff)] =
            Fp((uint32_t)(s_modulus[c_diff_idx] - core.c[c_diff_idx]));
        if (!core.is_setup) {
            core_row[CCOL(b_lt_diff)] =
                Fp((uint32_t)(s_modulus[b_diff_idx] - core.b[b_diff_idx]));
            bitwise.add_range(
                (uint32_t)(s_modulus[b_diff_idx] - core.b[b_diff_idx] - 1),
                (uint32_t)(s_modulus[c_diff_idx] - core.c[c_diff_idx] - 1));
        } else {
            core_row[CCOL(b_lt_diff)] = Fp::zero();
        }
        for (size_t i = 0; i < READ_LIMBS; i++) {
            uint32_t m = 0;
            if (i == b_diff_idx) {
                m = 1;
            } else if (i == c_diff_idx) {
                m = c_lt_mark;
            }
            core_row[CCOL(lt_marker) + i] = Fp(m);
        }

        bool is_eq = true;
        for (size_t i = 0; i < READ_LIMBS; i++) {
            core_row[CCOL(b) + i] = Fp((uint32_t)core.b[i]);
            core_row[CCOL(c) + i] = Fp((uint32_t)core.c[i]);
            if (core.b[i] != core.c[i] && is_eq) {
                is_eq = false;
                core_row[CCOL(eq_marker) + i] =
                    inv(Fp((uint32_t)core.b[i]) - Fp((uint32_t)core.c[i]));
            } else {
                core_row[CCOL(eq_marker) + i] = Fp::zero();
            }
        }
        core_row[CCOL(cmp_result)] = Fp((uint32_t)is_eq);
        core_row[CCOL(is_setup)] = Fp((uint32_t)core.is_setup);
        core_row[CCOL(is_valid)] = Fp::one();
#undef CCOL
    }
}

#define IS_EQ_LAUNCHER(LANES, LIMBS)                                                          \
    extern "C" int _modular_is_eq_tracegen_l##LANES(                                          \
        Fp *d_trace, size_t height, size_t rows_used, const uint8_t *d_records,               \
        size_t rec_stride, size_t rec_core_offset, const uint8_t *d_modulus_limbs,            \
        uint32_t *d_range_checker, size_t rc_bins, uint32_t *d_bitwise_lookup,               \
        size_t bitwise_num_bits, uint32_t pointer_max_bits,                                   \
        uint32_t timestamp_max_bits, cudaStream_t stream) {                                   \
        const int threads = 256;                                                              \
        const size_t want = (height + threads - 1) / threads;                                 \
        const int blocks = (int)(want < 1024 ? want : 1024);                                  \
        modular_is_eq_tracegen<2, LANES, LIMBS><<<blocks, threads, 0, stream>>>(              \
            d_trace, height, rows_used, d_records, rec_stride, rec_core_offset,               \
            d_modulus_limbs, d_range_checker, rc_bins, d_bitwise_lookup, bitwise_num_bits,   \
            pointer_max_bits, timestamp_max_bits);                                            \
        return CHECK_KERNEL();                                                                \
    }

IS_EQ_LAUNCHER(8, 32)  // 32-limb moduli (8 blocks of 4 bytes)
IS_EQ_LAUNCHER(12, 48) // 48-limb moduli
