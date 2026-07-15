// GPU tracegen for ModularIsEqual chips (IsEqualMod core + Rv64IsEqualModU16 adapter).
// One thread per row; transliterates ModularIsEqualFiller::fill_trace_row and
// Rv64IsEqualModU16AdapterFiller::fill_trace_row.
#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv-adapters/vec_heap.cuh" // brings riscv helpers (ptr_bound_from_high_u16, U16_BITS)
#include "system/memory/params.cuh"

#include <cstdint>

// ---- column structs (mirror the Rust #[repr(C)] AlignedBorrow layouts) ----

template <typename T, size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64IsEqualModU16AdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rs_val[NUM_READS][RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> heap_read_aux[NUM_READS][BLOCKS_PER_READ];

    T rd_ptr;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
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

template <size_t NUM_READS, size_t BLOCKS_PER_READ> struct Rv64IsEqualModU16AdapterRecord {
    uint32_t from_pc;
    uint32_t timestamp;

    uint32_t rs_ptr[NUM_READS];
    uint32_t rs_val[NUM_READS];
    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord heap_read_aux[NUM_READS][BLOCKS_PER_READ];

    uint32_t rd_ptr;
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

template <size_t READ_LIMBS> struct ModularIsEqualRecord {
    uint8_t is_setup; // bool
    uint16_t b[READ_LIMBS];
    uint16_t c[READ_LIMBS];
};

// run_unsigned_less_than: returns diff index (or READ_LIMBS when equal); lt result via out param.
template <size_t READ_LIMBS>
__device__ inline size_t
unsigned_less_than(const uint16_t (&x)[READ_LIMBS], const uint16_t *y, bool &lt) {
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
    const uint16_t *__restrict__ d_modulus_limbs, // READ_LIMBS u16 values
    uint32_t *__restrict__ d_range_checker,
    size_t rc_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits) {
    using AdapterColsU8 = Rv64IsEqualModU16AdapterCols<uint8_t, NUM_READS, BLOCKS_PER_READ>;
    using CoreColsU8 = ModularIsEqualCoreCols<uint8_t, READ_LIMBS>;
    constexpr size_t ADAPTER_WIDTH = sizeof(AdapterColsU8);
    constexpr size_t WIDTH = ADAPTER_WIDTH + sizeof(CoreColsU8);

    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t nthreads = gridDim.x * (size_t)blockDim.x;
    VariableRangeChecker rc(d_range_checker, rc_bins);
    MemoryAuxColsFactory mem_helper(rc, timestamp_max_bits);

    __shared__ uint16_t s_modulus[READ_LIMBS];
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
        const auto &rec = *(const Rv64IsEqualModU16AdapterRecord<NUM_READS, BLOCKS_PER_READ> *)
                              rec_bytes;
        const auto &core =
            *(const ModularIsEqualRecord<READ_LIMBS> *)(rec_bytes + rec_core_offset);

        // ---- adapter columns (mirror Rv64IsEqualModU16AdapterFiller) ----
#define ACOL(field) (offsetof(AdapterColsU8, field))
        for (size_t i = 0; i < NUM_READS; i++) {
            rc.add_count(
                ptr_bound_from_high_u16(uint16_t(rec.rs_val[i] >> U16_BITS), pointer_max_bits),
                U16_BITS);
        }
        uint32_t timestamp = rec.timestamp + (NUM_READS + NUM_READS * BLOCKS_PER_READ) + 1;

        // write aux (reverse timestamp order)
        timestamp--;
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++)
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
            slice[ACOL(rs_val[i]) + 0] = Fp(rec.rs_val[i] & 0xffffu);
            slice[ACOL(rs_val[i]) + 1] = Fp(rec.rs_val[i] >> U16_BITS);
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
            rc.add_count((uint32_t)(s_modulus[b_diff_idx] - core.b[b_diff_idx] - 1), U16_BITS);
            rc.add_count((uint32_t)(s_modulus[c_diff_idx] - core.c[c_diff_idx] - 1), U16_BITS);
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
        size_t rec_stride, size_t rec_core_offset, const uint16_t *d_modulus_limbs,           \
        uint32_t *d_range_checker, size_t rc_bins, uint32_t pointer_max_bits,                 \
        uint32_t timestamp_max_bits, cudaStream_t stream) {                                   \
        const int threads = 256;                                                              \
        const size_t want = (height + threads - 1) / threads;                                 \
        const int blocks = (int)(want < 1024 ? want : 1024);                                  \
        modular_is_eq_tracegen<2, LANES, LIMBS><<<blocks, threads, 0, stream>>>(              \
            d_trace, height, rows_used, d_records, rec_stride, rec_core_offset,               \
            d_modulus_limbs, d_range_checker, rc_bins, pointer_max_bits, timestamp_max_bits); \
        return CHECK_KERNEL();                                                                \
    }

IS_EQ_LAUNCHER(4, 16)  // 32-limb moduli (u16 lanes = 4 blocks, 16 u16 limbs)
IS_EQ_LAUNCHER(6, 24)  // 48-limb moduli
