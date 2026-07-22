#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/params.cuh"
#include <cub/device/device_scan.cuh>
#include <cstddef>
#include <cstdint>

/// Record representing a memory chunk with BLOCKS sub-blocks of cells.
/// Matches the Rust-side repr(C) `PersistentBoundaryRecord` layout.
///
/// Note on uint32_t encoding: only `values` stores Montgomery-encoded BabyBear
/// field elements (Fp::asRaw()). All other fields (`address_space`,
/// `ptr`, `is_dirty`, `timestamps`) are plain integers.
template <size_t CHUNK, size_t BLOCKS> struct MemoryInventoryRecord {
    uint32_t address_space; // plain integer
    uint32_t ptr;           // plain integer (address-space pointer)
    /// Whether some covered block was *written* during execution (0/1), tracked by
    /// preflight and carried in the input records; the merge kernel ORs the merged
    /// blocks' bits. Not consumed on device yet.
    uint32_t is_dirty;
    uint32_t timestamps[BLOCKS]; // plain integers
    uint32_t values[CHUNK];      // Montgomery-encoded Fp values (Fp::asRaw())
};

// Input records are one memory-bus message (`BLOCK_FE_WIDTH` cells, one
// timestamp). The merge kernel groups `BLOCKS_PER_LEAF` of them per merkle
// leaf (= `DIGEST_WIDTH` cells, `BLOCKS_PER_LEAF` timestamps).
using InRec = MemoryInventoryRecord<BLOCK_FE_WIDTH, 1>;
using OutRec = MemoryInventoryRecord<DIGEST_WIDTH, BLOCKS_PER_LEAF>;

__device__ inline bool same_output_block(
    InRec const *in,
    size_t lhs_idx,
    size_t rhs_idx
) {
    uint32_t lhs_as = in[lhs_idx].address_space;
    uint32_t rhs_as = in[rhs_idx].address_space;
    if (lhs_as != rhs_as) {
        return false;
    }
    return (in[lhs_idx].ptr / DIGEST_WIDTH) == (in[rhs_idx].ptr / DIGEST_WIDTH);
}

/// Read initial memory values for a Merkle leaf and convert them to Montgomery-encoded
/// field elements. The output values must be in Montgomery form because they are
/// stored directly into MemoryInventoryRecord.values, which boundary.cu later
/// reads via FpArray::from_raw_array (a raw copy that assumes Montgomery encoding).
///
/// `ptr` is an address-space pointer:
/// - DEFERRAL_AS: pointer into F cells; initial memory is already raw Montgomery Fp.
/// - Non-deferral ASes: pointer into u16 cells; initial memory is little-endian bytes.
__device__ inline void read_initial_leaf(
    uint32_t *out_values, // Montgomery-encoded Fp values
    uint8_t const *const *initial_mem,
    uint32_t address_space,
    uint32_t ptr
) {
    uint32_t addr_space_idx = address_space - 1;
    uint8_t const *mem = initial_mem[addr_space_idx];
    if (!mem) {
        #pragma unroll
        for (int i = 0; i < DIGEST_WIDTH; ++i) {
            out_values[i] = 0;
        }
        return;
    }
    if (address_space == DEFERRAL_AS) {
        // DEFERRAL_AS stores F cells directly, already raw Montgomery u32.
        uint32_t const *cells = reinterpret_cast<uint32_t const *>(mem) + ptr;
        #pragma unroll
        for (int i = 0; i < DIGEST_WIDTH; ++i) {
            out_values[i] = cells[i];
        }
    } else {
        // u16 cells, little-endian. Each cell occupies `U16_CELL_SIZE` bytes at
        // byte offset `U16_CELL_SIZE * (ptr + i)`.
        size_t base = static_cast<size_t>(ptr) * U16_CELL_SIZE;
        #pragma unroll
        for (int i = 0; i < DIGEST_WIDTH; ++i) {
            out_values[i] = Fp(u16_from_bytes_le(mem + base + U16_CELL_SIZE * i)).asRaw();
        }
    }
}

__global__ void cukernel_build_candidates(
    InRec const *in,
    size_t in_num_records,
    uint8_t const *const *initial_mem,
    OutRec *tmp_out,
    uint32_t *flags
) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= in_num_records) {
        return;
    }
    if (row_idx != 0 && same_output_block(in, row_idx - 1, row_idx)) {
        flags[row_idx] = 0;
        return;
    }
    flags[row_idx] = 1;

    OutRec rec{};
    rec.address_space = in[row_idx].address_space;
    rec.ptr = (in[row_idx].ptr / DIGEST_WIDTH) * DIGEST_WIDTH;
    rec.is_dirty = in[row_idx].is_dirty;
    #pragma unroll
    for (size_t i = 0; i < BLOCKS_PER_LEAF; ++i) {
        rec.timestamps[i] = 0;
    }

    // Fill all values with Montgomery-encoded initial memory
    read_initial_leaf(rec.values, initial_mem, rec.address_space, rec.ptr);

    // Overwrite touched block's values (already Montgomery-encoded in input records)
    uint32_t block_idx = (in[row_idx].ptr % DIGEST_WIDTH) / BLOCK_FE_WIDTH;
    #pragma unroll
    for (int i = 0; i < BLOCK_FE_WIDTH; ++i) {
        rec.values[block_idx * BLOCK_FE_WIDTH + i] = in[row_idx].values[i];
    }
    rec.timestamps[block_idx] = in[row_idx].timestamps[0];

    // If two input records fall in the same chunk, overwrite the second block too
    if (row_idx + 1 < in_num_records && same_output_block(in, row_idx, row_idx + 1)) {
        uint32_t block_idx2 = (in[row_idx + 1].ptr % DIGEST_WIDTH) / BLOCK_FE_WIDTH;
        #pragma unroll
        for (int i = 0; i < BLOCK_FE_WIDTH; ++i) {
            rec.values[block_idx2 * BLOCK_FE_WIDTH + i] = in[row_idx + 1].values[i];
        }
        rec.timestamps[block_idx2] = in[row_idx + 1].timestamps[0];
        rec.is_dirty |= in[row_idx + 1].is_dirty;
    }

    tmp_out[row_idx] = rec;
}

__global__ void cukernel_scatter_compact(
    OutRec const *tmp_out,
    uint32_t const *flags,
    uint32_t const *positions,
    size_t n,
    OutRec *out,
    size_t *out_num_records
) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= n) return;
    if (flags[row_idx]) {
        uint32_t out_idx = positions[row_idx];
        out[out_idx] = tmp_out[row_idx];
    }
    if (row_idx == n - 1) {
        *out_num_records = static_cast<size_t>(positions[row_idx] + flags[row_idx]);
    }
}

extern "C" int _inventory_merge_records(
    uint32_t const *d_in_records,
    size_t in_num_records,
    uint8_t const *const *d_initial_mem,
    uint32_t *d_tmp_records,
    uint32_t *d_out_records,
    uint32_t *d_flags,
    uint32_t *d_positions,
    void *d_temp_storage,
    size_t temp_storage_bytes,
    size_t *out_num_records,
    cudaStream_t stream
) {
    auto [grid, block] = kernel_launch_params(in_num_records);
    InRec const *in = reinterpret_cast<InRec const *>(d_in_records);
    OutRec *tmp_out = reinterpret_cast<OutRec *>(d_tmp_records);
    OutRec *out = reinterpret_cast<OutRec *>(d_out_records);

    cukernel_build_candidates<<<grid, block, 0, stream>>>(
        in,
        in_num_records,
        d_initial_mem,
        tmp_out,
        d_flags
    );
    if (int err = CHECK_KERNEL(); err) {
        return err;
    }

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_flags,
        d_positions,
        in_num_records,
        stream
    );
    if (int err = CHECK_KERNEL(); err) {
        return err;
    }

    cukernel_scatter_compact<<<grid, block, 0, stream>>>(
        tmp_out,
        d_flags,
        d_positions,
        in_num_records,
        out,
        out_num_records
    );
    return CHECK_KERNEL();
}

extern "C" int _inventory_merge_records_get_temp_bytes(
    uint32_t *d_flags,
    size_t in_num_records,
    size_t *h_temp_bytes_out,
    cudaStream_t stream
) {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_bytes,
        d_flags,
        d_flags,
        in_num_records,
        stream
    );
    *h_temp_bytes_out = temp_bytes;
    return CHECK_KERNEL();
}

/// Width in u32 words of one Merkle touched-block record:
/// (address_space, ptr, timestamp, values[DIGEST_WIDTH]).
/// Must match MERKLE_TOUCHED_BLOCK_WIDTH on the Rust side.
inline constexpr uint32_t MERKLE_REC_WIDTH = 3 + DIGEST_WIDTH;

/// Converts merged inventory records (boundary layout) into Merkle
/// touched-block records: drops the second timestamp, taking the max.
/// Values stay Montgomery-encoded in both layouts.
__global__ void inventory_to_merkle_records_kernel(
    uint32_t const *out_records,
    size_t num_records,
    uint32_t *merkle_records
) {
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < num_records;
         i += stride) {
        OutRec const &r = ((OutRec const *)out_records)[i];
        uint32_t *dst = merkle_records + i * MERKLE_REC_WIDTH;
        dst[0] = r.address_space;
        dst[1] = r.ptr;
        dst[2] = max(r.timestamps[0], r.timestamps[1]);
#pragma unroll
        for (int j = 0; j < DIGEST_WIDTH; ++j) {
            dst[3 + j] = r.values[j];
        }
    }
}

extern "C" int _inventory_to_merkle_records(
    const uint32_t *d_out_records,
    size_t num_records,
    uint32_t *d_merkle_records,
    cudaStream_t stream
) {
    if (num_records == 0) {
        return 0;
    }
    auto [grid, block] = grid_stride_launch_params(num_records, 256, 1024);
    inventory_to_merkle_records_kernel<<<grid, block, 0, stream>>>(
        d_out_records, num_records, d_merkle_records);
    return CHECK_KERNEL();
}
