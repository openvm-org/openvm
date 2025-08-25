#include "launcher.cuh"
#include "primitives/fp_array.cuh"
#include "primitives/less_than.cuh"
#include "primitives/shared_buffer.cuh"
#include "primitives/trace_access.h"
#include <cassert>

static const size_t PERSISTENT_CHUNK = 8;
static const size_t VOLATILE_CHUNK = 1;

template <size_t CHUNK> struct BoundaryRecord {
    uint32_t address_space;
    uint32_t ptr;
    uint32_t timestamp;
    uint32_t values[CHUNK];
};

template <typename T> struct PersistentBoundaryCols {
    T expand_direction;
    T address_space;
    T leaf_label;
    T values[PERSISTENT_CHUNK];
    T hash[PERSISTENT_CHUNK];
    T timestamp;
};

static const size_t ADDR_ELTS = 2;
static const size_t NUM_AS_LIMBS = 1;

template <typename T> struct VolatileBoundaryCols {
    T address_space_limbs[NUM_AS_LIMBS];
    T pointer_limbs[AUX_LEN];
    T initial_data;
    T final_data;
    T final_timestamp;
    T is_valid;
    LessThanArrayAuxCols<T, ADDR_ELTS, AUX_LEN> addr_lt_aux;
};

__global__ void cukernel_persistent_boundary_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    uint8_t const *const *initial_mem,
    BoundaryRecord<PERSISTENT_CHUNK> *records,
    size_t num_records,
    FpArray<16> *poseidon2_buffer,
    uint32_t *poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t record_idx = row_idx / 2;
    RowSlice row = RowSlice(trace + row_idx, height);

    if (record_idx < num_records) {
        BoundaryRecord<PERSISTENT_CHUNK> record = records[record_idx];
        Poseidon2Buffer poseidon2(poseidon2_buffer, poseidon2_buffer_idx, poseidon2_capacity);
        COL_WRITE_VALUE(row, PersistentBoundaryCols, address_space, record.address_space);
        COL_WRITE_VALUE(row, PersistentBoundaryCols, leaf_label, record.ptr / PERSISTENT_CHUNK);
        if (row_idx % 2 == 0) {
            // TODO better address space handling
            FpArray<8> init_values =
                record.address_space == 4
                    ? FpArray<8>::from_raw_array(
                          reinterpret_cast<uint32_t const *>(
                              initial_mem[record.address_space - 1]
                          ) +
                          record.ptr
                      )
                    : FpArray<8>::from_u8_array(initial_mem[record.address_space - 1] + record.ptr);
            FpArray<8> init_hash = poseidon2.hash_and_record(init_values);
            COL_WRITE_VALUE(row, PersistentBoundaryCols, expand_direction, Fp::one());
            COL_WRITE_VALUE(row, PersistentBoundaryCols, timestamp, Fp::zero());
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, values, reinterpret_cast<Fp const *>(init_values.v)
            );
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, hash, reinterpret_cast<Fp const *>(init_hash.v)
            );
        } else {
            FpArray<8> final_values = FpArray<8>::from_raw_array(record.values);
            FpArray<8> final_hash = poseidon2.hash_and_record(final_values);
            COL_WRITE_VALUE(row, PersistentBoundaryCols, expand_direction, Fp::neg_one());
            COL_WRITE_VALUE(row, PersistentBoundaryCols, timestamp, record.timestamp);
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, values, reinterpret_cast<Fp const *>(final_values.v)
            );
            COL_WRITE_ARRAY(
                row, PersistentBoundaryCols, hash, reinterpret_cast<Fp const *>(final_hash.v)
            );
        }
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void cukernel_volatile_boundary_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    BoundaryRecord<VOLATILE_CHUNK> const *records,
    size_t num_records,
    uint32_t *range_checker,
    size_t range_checker_num_bins,
    size_t as_max_bits,
    size_t ptr_max_bits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row = RowSlice(trace + idx, height);
    VariableRangeChecker rc(range_checker, range_checker_num_bins);
    assert(idx < height);

    if (idx < num_records) {
        if (idx == num_records - 1) {
            // For the sake of always filling `addr_lt_aux`
            row.fill_zero(0, width);
        }
        BoundaryRecord<VOLATILE_CHUNK> record = records[idx];
        rc.decompose(
            record.address_space,
            as_max_bits,
            row.slice_from(COL_INDEX(VolatileBoundaryCols, address_space_limbs)),
            NUM_AS_LIMBS
        );
        rc.decompose(
            record.ptr,
            ptr_max_bits,
            row.slice_from(COL_INDEX(VolatileBoundaryCols, pointer_limbs)),
            AUX_LEN
        );
        COL_WRITE_VALUE(row, VolatileBoundaryCols, initial_data, Fp::zero());
        COL_WRITE_VALUE(row, VolatileBoundaryCols, final_data, record.values[0]);
        COL_WRITE_VALUE(row, VolatileBoundaryCols, final_timestamp, record.timestamp);
        COL_WRITE_VALUE(row, VolatileBoundaryCols, is_valid, Fp::one());

        if (idx != num_records - 1) {
            BoundaryRecord<VOLATILE_CHUNK> next_record = records[idx + 1];
            uint32_t curr[ADDR_ELTS] = {record.address_space, record.ptr};
            uint32_t next[ADDR_ELTS] = {next_record.address_space, next_record.ptr};
            IsLessThanArray::generate_subrow(
                rc,
                max(as_max_bits, ptr_max_bits),
                FpArray<ADDR_ELTS>::from_u32_array(curr),
                FpArray<ADDR_ELTS>::from_u32_array(next),
                AUX_LEN,
                RowSlice(row.slice_from(COL_INDEX(VolatileBoundaryCols, addr_lt_aux.diff_marker))),
                row.slice_from(COL_INDEX(VolatileBoundaryCols, addr_lt_aux.diff_inv)).ptr,
                RowSlice(row.slice_from(COL_INDEX(VolatileBoundaryCols, addr_lt_aux.lt_decomp))),
                nullptr
            );
        }
    } else {
        row.fill_zero(0, width);
    }

    if (idx == height - 1 && num_records > 0) {
        uint32_t zeros[ADDR_ELTS] = {0, 0};
        FpArray<ADDR_ELTS> zeros_fp = FpArray<ADDR_ELTS>::from_raw_array(zeros);
        IsLessThanArray::generate_subrow(
            rc,
            max(as_max_bits, ptr_max_bits),
            zeros_fp,
            zeros_fp,
            AUX_LEN,
            RowSlice(row.slice_from(COL_INDEX(VolatileBoundaryCols, addr_lt_aux.diff_marker))),
            row.slice_from(COL_INDEX(VolatileBoundaryCols, addr_lt_aux.diff_inv)).ptr,
            RowSlice(row.slice_from(COL_INDEX(VolatileBoundaryCols, addr_lt_aux.lt_decomp))),
            nullptr
        );
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
    size_t poseidon2_capacity
) {
    auto [grid, block] = kernel_launch_params(height);
    BoundaryRecord<PERSISTENT_CHUNK> *d_records =
        reinterpret_cast<BoundaryRecord<PERSISTENT_CHUNK> *>(d_raw_records);
    FpArray<16> *d_poseidon2_buffer = reinterpret_cast<FpArray<16> *>(d_poseidon2_raw_buffer);
    cukernel_persistent_boundary_tracegen<<<grid, block>>>(
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
    return cudaGetLastError();
}

extern "C" int _volatile_boundary_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint32_t const *d_raw_records,
    size_t num_records,
    uint32_t *d_range_checker,
    size_t range_checker_num_bins,
    size_t as_max_bits,
    size_t ptr_max_bits
) {
    auto [grid, block] = kernel_launch_params(height, 512);
    auto d_records = reinterpret_cast<BoundaryRecord<VOLATILE_CHUNK> const *>(d_raw_records);
    cukernel_volatile_boundary_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        num_records,
        d_range_checker,
        range_checker_num_bins,
        as_max_bits,
        ptr_max_bits
    );
    return cudaGetLastError();
}
