#include "launcher.cuh"
#include "primitives/fp_array.cuh"
#include "primitives/shared_buffer.cuh"
#include "primitives/trace_access.h"
#include "system/memory/params.cuh"

template <size_t CHUNK, size_t BLOCKS> struct BoundaryRecord {
    uint32_t address_space;
    uint32_t ptr;
    uint32_t timestamps[BLOCKS];
    uint32_t values[CHUNK]; // Montgomery-encoded Fp values stored as raw u32
};

template <typename T> struct PersistentBoundaryCols {
    T expand_direction;
    T address_space;
    T leaf_label;
    T values[DIGEST_WIDTH];
    T hash[DIGEST_WIDTH];
    T timestamps[BLOCKS_PER_LEAF];
};

__global__ void cukernel_persistent_boundary_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    uint8_t const *const *initial_mem,
    BoundaryRecord<DIGEST_WIDTH, BLOCKS_PER_LEAF> *records,
    size_t num_records,
    FpArray<POSEIDON2_WIDTH> *poseidon2_buffer,
    uint32_t *poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t record_idx = row_idx / 2;
    RowSlice row = RowSlice(trace + row_idx, height);

    if (record_idx < num_records) {
        BoundaryRecord<DIGEST_WIDTH, BLOCKS_PER_LEAF> record = records[record_idx];
        Poseidon2Buffer poseidon2(poseidon2_buffer, poseidon2_buffer_idx, poseidon2_capacity);
        COL_WRITE_VALUE(row, PersistentBoundaryCols, address_space, record.address_space);
        COL_WRITE_VALUE(row, PersistentBoundaryCols, leaf_label, record.ptr / DIGEST_WIDTH);
        if (row_idx % 2 == 0) {
            FpArray<DIGEST_WIDTH> init_values;
            uint32_t addr_space_idx = record.address_space - 1;
            if (initial_mem[addr_space_idx]) {
                // `record.ptr` is the AS-specific **cell index** (matches the
                // partition key the GPU inventory carries forward). Cell sizes:
                //   - DEFERRAL_AS: 4 bytes per cell (raw Fp).
                //   - AS 1/2/3 (u16-celled): 2 bytes per cell, little-endian.
                init_values =
                    record.address_space == DEFERRAL_AS
                        ? FpArray<DIGEST_WIDTH>::from_raw_array(
                            reinterpret_cast<uint32_t const *>(
                                initial_mem[addr_space_idx]
                            ) + record.ptr
                        )
                        : FpArray<DIGEST_WIDTH>::from_u16_le_array(
                            initial_mem[addr_space_idx] + U16_CELL_SIZE * record.ptr
                        );
            } else {
                #pragma unroll
                for (int i = 0; i < DIGEST_WIDTH; ++i) {
                    init_values.v[i] = 0;
                }
            }
            FpArray<DIGEST_WIDTH> init_hash = poseidon2.hash_and_record(init_values);
            COL_WRITE_VALUE(row, PersistentBoundaryCols, expand_direction, Fp::one());
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, values, reinterpret_cast<Fp const *>(init_values.v)
            );
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, hash, reinterpret_cast<Fp const *>(init_hash.v)
            );
            row.fill_zero(COL_INDEX(PersistentBoundaryCols, timestamps), BLOCKS_PER_LEAF);
        } else {
            // record.values are already Montgomery-encoded (see read_initial_chunk in inventory.cu)
            FpArray<DIGEST_WIDTH> final_values = FpArray<DIGEST_WIDTH>::from_raw_array(record.values);
            FpArray<DIGEST_WIDTH> final_hash = poseidon2.hash_and_record(final_values);
            COL_WRITE_VALUE(row, PersistentBoundaryCols, expand_direction, Fp::neg_one());
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, values, reinterpret_cast<Fp const *>(final_values.v)
            );
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, hash, reinterpret_cast<Fp const *>(final_hash.v)
            );
            COL_WRITE_ARRAY(row, PersistentBoundaryCols, timestamps, record.timestamps);
        }
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _persistent_boundary_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t const *const *d_initial_mem,
    uint32_t *d_raw_records,
    size_t num_records,
    Fp *d_poseidon2_raw_buffer,
    uint32_t *d_poseidon2_buffer_idx,
    size_t poseidon2_capacity,
    cudaStream_t stream
) {
    auto [grid, block] = kernel_launch_params(height);
    BoundaryRecord<DIGEST_WIDTH, BLOCKS_PER_LEAF> *d_records =
        reinterpret_cast<BoundaryRecord<DIGEST_WIDTH, BLOCKS_PER_LEAF> *>(d_raw_records);
    FpArray<POSEIDON2_WIDTH> *d_poseidon2_buffer =
        reinterpret_cast<FpArray<POSEIDON2_WIDTH> *>(d_poseidon2_raw_buffer);
    // poseidon2_capacity arrives from Rust in units of Fp elements; convert to record count.
    assert(
        poseidon2_capacity % POSEIDON2_WIDTH == 0
        && "poseidon2_capacity must be a multiple of POSEIDON2_WIDTH"
    );
    size_t poseidon2_record_capacity = poseidon2_capacity / POSEIDON2_WIDTH;
    cukernel_persistent_boundary_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_initial_mem,
        d_records,
        num_records,
        d_poseidon2_buffer,
        d_poseidon2_buffer_idx,
        poseidon2_record_capacity
    );
    return CHECK_KERNEL();
}
