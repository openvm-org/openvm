#include "launcher.cuh"
#include "system/records.cuh"
#include <cub/device/device_scan.cuh>
#include <cstddef>
#include <cstdint>

const uint32_t IN_BLOCK_SIZE = 4;
const uint32_t OUT_BLOCK_SIZE = 8;

using InRec = MemoryInventoryRecord<IN_BLOCK_SIZE, 1>;
using OutRec = MemoryInventoryRecord<OUT_BLOCK_SIZE, 2>;

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
    return (in[lhs_idx].ptr / OUT_BLOCK_SIZE) == (in[rhs_idx].ptr / OUT_BLOCK_SIZE);
}

__device__ inline uint32_t addr_space_cell_size(
    uint32_t const *addr_space_offsets,
    uint32_t addr_space_idx
) {
    return addr_space_offsets[addr_space_idx + 1] - addr_space_offsets[addr_space_idx];
}

__device__ inline void read_initial_chunk(
    uint32_t *out_values,
    uint8_t const *const *initial_mem,
    uint32_t const *addr_space_offsets,
    uint32_t address_space,
    uint32_t chunk_ptr
) {
    uint32_t addr_space_idx = address_space - 1;
    uint8_t const *mem = initial_mem[addr_space_idx];
    if (!mem) {
        for (int i = 0; i < OUT_BLOCK_SIZE; ++i) {
            out_values[i] = 0;
        }
        return;
    }
    uint32_t cell_size = addr_space_cell_size(addr_space_offsets, addr_space_idx);
    size_t byte_offset = static_cast<size_t>(chunk_ptr) * cell_size;
    for (int i = 0; i < OUT_BLOCK_SIZE; ++i) {
        size_t off = byte_offset + static_cast<size_t>(i) * cell_size;
        if (cell_size == 4) {
            out_values[i] = *reinterpret_cast<uint32_t const *>(mem + off);
        } else if (cell_size == 2) {
            out_values[i] = *reinterpret_cast<uint16_t const *>(mem + off);
        } else if (cell_size == 1) {
            out_values[i] = mem[off];
        } else {
            out_values[i] = 0;
        }
    }
}

__global__ void cukernel_build_candidates(
    InRec const *in,
    size_t in_num_records,
    uint8_t const *const *initial_mem,
    uint32_t const *addr_space_offsets,
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
    uint32_t chunk_ptr = (in[row_idx].ptr / OUT_BLOCK_SIZE) * OUT_BLOCK_SIZE;
    rec.ptr = chunk_ptr;
    rec.timestamps[0] = 0;
    rec.timestamps[1] = 0;

    read_initial_chunk(rec.values, initial_mem, addr_space_offsets, rec.address_space, chunk_ptr);

    uint32_t block_idx = (in[row_idx].ptr % OUT_BLOCK_SIZE) / IN_BLOCK_SIZE;
    for (int i = 0; i < IN_BLOCK_SIZE; ++i) {
        rec.values[block_idx * IN_BLOCK_SIZE + i] = in[row_idx].values[i];
    }
    rec.timestamps[block_idx] = in[row_idx].timestamps[0];

    if (row_idx + 1 < in_num_records && same_output_block(in, row_idx, row_idx + 1)) {
        uint32_t block_idx2 = (in[row_idx + 1].ptr % OUT_BLOCK_SIZE) / IN_BLOCK_SIZE;
        for (int i = 0; i < IN_BLOCK_SIZE; ++i) {
            rec.values[block_idx2 * IN_BLOCK_SIZE + i] = in[row_idx + 1].values[i];
        }
        rec.timestamps[block_idx2] = in[row_idx + 1].timestamps[0];
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
    uint32_t const *d_addr_space_offsets,
    uint32_t *d_tmp_records,
    uint32_t *d_out_records,
    uint32_t *d_flags,
    uint32_t *d_positions,
    void *d_temp_storage,
    size_t temp_storage_bytes,
    size_t *out_num_records
) {
    auto [grid, block] = kernel_launch_params(in_num_records);
    InRec const *in = reinterpret_cast<InRec const *>(d_in_records);
    OutRec *tmp_out = reinterpret_cast<OutRec *>(d_tmp_records);
    OutRec *out = reinterpret_cast<OutRec *>(d_out_records);

    cukernel_build_candidates<<<grid, block>>>(
        in,
        in_num_records,
        d_initial_mem,
        d_addr_space_offsets,
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
        cudaStreamPerThread
    );
    if (int err = CHECK_KERNEL(); err) {
        return err;
    }

    cukernel_scatter_compact<<<grid, block>>>(
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
    size_t *h_temp_bytes_out
) {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_bytes,
        d_flags,
        d_flags,
        in_num_records,
        cudaStreamPerThread
    );
    *h_temp_bytes_out = temp_bytes;
    return CHECK_KERNEL();
}
