#include "launcher.cuh"
#include "primitives/fp_array.cuh"
#include "primitives/shared_buffer.cuh"
#include "primitives/trace_access.h"

inline constexpr size_t CHUNK = 8;
inline constexpr size_t BLOCKS_PER_CHUNK = 2;

template <size_t CHUNK, size_t BLOCKS> struct MemoryBoundaryRecord {
    uint32_t address_space;
    uint32_t ptr;
    uint32_t timestamps[BLOCKS];
    uint32_t values[CHUNK]; // Montgomery-encoded Fp values stored as raw u32
};

template <typename T> struct MemoryBoundaryCols {
    T expand_direction;
    T address_space;
    T leaf_label;
    T values[CHUNK];
    T hash[CHUNK];
    T timestamps[BLOCKS_PER_CHUNK];
};

__global__ void cukernel_boundary_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    uint8_t const *const *initial_mem,
    MemoryBoundaryRecord<CHUNK, BLOCKS_PER_CHUNK> *records,
    size_t num_records,
    FpArray<16> *poseidon2_buffer,
    uint32_t *poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t record_idx = row_idx / 2;
    RowSlice row = RowSlice(trace + row_idx, height);

    if (record_idx < num_records) {
        MemoryBoundaryRecord<CHUNK, BLOCKS_PER_CHUNK> record = records[record_idx];
        Poseidon2Buffer poseidon2(poseidon2_buffer, poseidon2_buffer_idx, poseidon2_capacity);
        COL_WRITE_VALUE(row, MemoryBoundaryCols, address_space, record.address_space);
        COL_WRITE_VALUE(row, MemoryBoundaryCols, leaf_label, record.ptr / CHUNK);
        if (row_idx % 2 == 0) {
            // TODO better address space handling
            FpArray<8> init_values;
            if (initial_mem[record.address_space - 1]) {
                init_values = FpArray<8>::from_u8_array(initial_mem[record.address_space - 1] + record.ptr);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    init_values.v[i] = 0;
                }
            }
            FpArray<8> init_hash = poseidon2.hash_and_record(init_values);
            COL_WRITE_VALUE(row, MemoryBoundaryCols, expand_direction, Fp::one());
            COL_WRITE_ARRAY(
                row, MemoryBoundaryCols, values, reinterpret_cast<Fp const *>(init_values.v)
            );
            COL_WRITE_ARRAY(
                row, MemoryBoundaryCols, hash, reinterpret_cast<Fp const *>(init_hash.v)
            );
            row.fill_zero(COL_INDEX(MemoryBoundaryCols, timestamps), BLOCKS_PER_CHUNK);
        } else {
            // record.values are already Montgomery-encoded (see read_initial_chunk in inventory.cu)
            FpArray<8> final_values = FpArray<8>::from_raw_array(record.values);
            FpArray<8> final_hash = poseidon2.hash_and_record(final_values);
            COL_WRITE_VALUE(row, MemoryBoundaryCols, expand_direction, Fp::neg_one());
            COL_WRITE_ARRAY(
                row, MemoryBoundaryCols, values, reinterpret_cast<Fp const *>(final_values.v)
            );
            COL_WRITE_ARRAY(
                row, MemoryBoundaryCols, hash, reinterpret_cast<Fp const *>(final_hash.v)
            );
            COL_WRITE_ARRAY(row, MemoryBoundaryCols, timestamps, record.timestamps);
        }
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _boundary_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t const *const *d_initial_mem,
    uint32_t *d_raw_records,
    size_t num_records,
    Fp *d_poseidon2_raw_buffer,
    uint32_t *d_poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    auto [grid, block] = kernel_launch_params(height);
    MemoryBoundaryRecord<CHUNK, BLOCKS_PER_CHUNK> *d_records =
        reinterpret_cast<MemoryBoundaryRecord<CHUNK, BLOCKS_PER_CHUNK> *>(d_raw_records);
    FpArray<16> *d_poseidon2_buffer = reinterpret_cast<FpArray<16> *>(d_poseidon2_raw_buffer);
    cukernel_boundary_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_initial_mem,
        d_records,
        num_records,
        d_poseidon2_buffer,
        d_poseidon2_buffer_idx,
        poseidon2_capacity
    );
    return CHECK_KERNEL();
}
