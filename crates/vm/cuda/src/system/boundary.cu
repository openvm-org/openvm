#include "launcher.cuh"
#include "primitives/fp_array.cuh"
#include "primitives/shared_buffer.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/params.cuh"

template <size_t CHUNK, size_t BLOCKS> struct BoundaryRecord {
    uint32_t address_space;
    // AS-native pointer to the first cell of this Merkle leaf.
    uint32_t ptr;
    // Whether some block of the leaf was *written* during execution (0/1), tracked by
    // preflight and merged by inventory.cu. Not consumed yet.
    uint32_t is_dirty;
    uint32_t timestamps[BLOCKS];
    uint32_t values[CHUNK]; // Montgomery-encoded Fp values stored as raw u32
};

template <typename T> struct PersistentBoundaryCols {
    T is_valid;
    T is_dirty;
    T address_space;
    T leaf_label;
    T initial_values[DIGEST_WIDTH];
    T final_values[DIGEST_WIDTH];
    T initial_hash[DIGEST_WIDTH];
    T final_hash[DIGEST_WIDTH];
    T final_timestamps[BLOCKS_PER_LEAF];
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
    RowSlice row = RowSlice(trace + row_idx, height);

    if (row_idx < num_records) {
        BoundaryRecord<DIGEST_WIDTH, BLOCKS_PER_LEAF> record = records[row_idx];
        Poseidon2Buffer poseidon2(poseidon2_buffer, poseidon2_buffer_idx, poseidon2_capacity);
        COL_WRITE_VALUE(row, PersistentBoundaryCols, is_valid, Fp::one());
        // TODO(follow-up): set real dirtiness (`final_values != init_values`) once
        // MemoryMerkleAir supports skipping clean leaves in the final expansion. Until
        // then every touched leaf is treated as dirty, matching the CPU tracegen.
        COL_WRITE_VALUE(row, PersistentBoundaryCols, is_dirty, Fp::one());
        COL_WRITE_VALUE(row, PersistentBoundaryCols, address_space, record.address_space);
        COL_WRITE_VALUE(
            row,
            PersistentBoundaryCols,
            leaf_label,
            record.ptr / DIGEST_WIDTH
        );

        FpArray<DIGEST_WIDTH> init_values;
        uint32_t addr_space_idx = record.address_space - 1;
        if (initial_mem[addr_space_idx]) {
            // `ptr` is an address-space pointer:
            //   - DEFERRAL_AS: pointer into F cells; initial memory is already raw Montgomery Fp.
            //   - Non-deferral ASes: pointer into u16 cells; initial memory is little-endian bytes,
            //     so convert the pointer to a byte offset with `U16_CELL_SIZE`.
            if (record.address_space == DEFERRAL_AS) {
                init_values = FpArray<DIGEST_WIDTH>::from_raw_array(
                    reinterpret_cast<uint32_t const *>(initial_mem[addr_space_idx]) +
                    record.ptr
                );
            } else {
                uint8_t const *bytes =
                    initial_mem[addr_space_idx] + U16_CELL_SIZE * record.ptr;
                uint16_t cells[DIGEST_WIDTH];
                #pragma unroll
                for (int i = 0; i < DIGEST_WIDTH; ++i) {
                    cells[i] = u16_from_bytes_le(bytes + U16_CELL_SIZE * i);
                }
                init_values = FpArray<DIGEST_WIDTH>::from_u16_array(cells);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DIGEST_WIDTH; ++i) {
                init_values.v[i] = 0;
            }
        }
        FpArray<DIGEST_WIDTH> init_hash = poseidon2.hash_and_record(init_values);
        COL_WRITE_ARRAY(
            row, PersistentBoundaryCols, initial_values, reinterpret_cast<Fp const *>(init_values.v)
        );
        COL_WRITE_ARRAY(
            row, PersistentBoundaryCols, initial_hash, reinterpret_cast<Fp const *>(init_hash.v)
        );

        // record.values are already Montgomery-encoded (see read_initial_chunk in inventory.cu)
        FpArray<DIGEST_WIDTH> final_values = FpArray<DIGEST_WIDTH>::from_raw_array(record.values);
        FpArray<DIGEST_WIDTH> final_hash = poseidon2.hash_and_record(final_values);
        COL_WRITE_ARRAY(
            row, PersistentBoundaryCols, final_values, reinterpret_cast<Fp const *>(final_values.v)
        );
        COL_WRITE_ARRAY(
            row, PersistentBoundaryCols, final_hash, reinterpret_cast<Fp const *>(final_hash.v)
        );
        COL_WRITE_ARRAY(row, PersistentBoundaryCols, final_timestamps, record.timestamps);
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
