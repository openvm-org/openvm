#include "primitives/less_than.cuh"
#include "primitives/trace_access.h"
#include <algorithm>

inline __device__ uint32_t next_power_of_two_or_zero(uint32_t x) {
    return x ? (1u << (32 - __clz(x - 1))) : 0;
}

template <typename T, size_t N> struct AccessAdapterCols {
    T is_valid;
    T is_split;
    T address_space;
    T pointer;
    T values[N];
    T left_timestamp;
    T right_timestamp;
    T is_right_larger;
    T lt_aux[AUX_LEN];
};

auto constexpr MERGE_AND_NOT_SPLIT_FLAG = 1u << 31;

struct AccessAdapterRecordHeader {
    uint32_t timestamp_and_mask;
    uint32_t address_space;
    uint32_t pointer;
    // TODO: these three are easily mergeable into a single u32
    // Sync these changes with the openvm ones
    uint32_t block_size;
    uint32_t lowest_block_size;
    uint32_t type_size;
};

template <size_t N> struct Cols {
    template <typename T> using type = AccessAdapterCols<T, N>;
};

template <size_t N>
__device__ void _fill_trace_row(
    Fp *d_trace,
    uint32_t row_idx,
    bool is_split,
    uint32_t address_space,
    uint32_t pointer,
    uint8_t const *data,
    uint32_t left_timestamp,
    uint32_t right_timestamp,
    uint32_t type_size,
    uint32_t unpadded_trace_height,
    uint32_t *d_range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t const padded_height = next_power_of_two_or_zero(unpadded_trace_height);
    if (auto back_idx = padded_height - 1 - row_idx; back_idx >= unpadded_trace_height) {
        RowSlice row(d_trace + back_idx, padded_height);
        row.fill_zero(0, sizeof(AccessAdapterCols<uint8_t, N>));
    }
    RowSlice row(d_trace + row_idx, padded_height);

    COL_WRITE_VALUE(row, typename Cols<N>::template type, is_valid, Fp::one());
    COL_WRITE_VALUE(row, typename Cols<N>::template type, is_split, is_split);
    COL_WRITE_VALUE(row, typename Cols<N>::template type, address_space, address_space);
    COL_WRITE_VALUE(row, typename Cols<N>::template type, pointer, pointer);
    if (type_size == 1) {
        COL_WRITE_ARRAY(row, typename Cols<N>::template type, values, data);
    } else if (type_size == 4) {
        COL_WRITE_ARRAY(
            row, typename Cols<N>::template type, values, reinterpret_cast<Fp const *>(data)
        );
    } else {
        assert(false);
    }
    COL_WRITE_VALUE(row, typename Cols<N>::template type, left_timestamp, left_timestamp);
    COL_WRITE_VALUE(row, typename Cols<N>::template type, right_timestamp, right_timestamp);
    COL_WRITE_VALUE(
        row, typename Cols<N>::template type, is_right_larger, right_timestamp > left_timestamp
    );
    VariableRangeChecker range_checker(d_range_checker, range_checker_bins);
    Fp *out = &row[COL_INDEX(typename Cols<N>::template type, is_right_larger)];
    Fp *decomp = &row[COL_INDEX(typename Cols<N>::template type, lt_aux)];

    IsLessThan::generate_subrow(
        range_checker,
        timestamp_max_bits,
        left_timestamp,
        right_timestamp,
        AUX_LEN,
        RowSlice(decomp, padded_height),
        out
    );
}

template <typename... Args> __device__ void fill_trace_row(size_t const n, Args &&...args) {
    switch (n) {
    case 2:
        _fill_trace_row<2>(args...);
        break;
    case 4:
        _fill_trace_row<4>(args...);
        break;
    case 8:
        _fill_trace_row<8>(args...);
        break;
    case 16:
        _fill_trace_row<16>(args...);
        break;
    case 32:
        _fill_trace_row<32>(args...);
        break;
    default:
        assert(false);
    }
}

__device__ void assert_widths(size_t const *const widths, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        switch (i) {
        case 0:
            assert(widths[i] == sizeof(AccessAdapterCols<uint8_t, 2>));
            break;
        case 1:
            assert(widths[i] == sizeof(AccessAdapterCols<uint8_t, 4>));
            break;
        case 2:
            assert(widths[i] == sizeof(AccessAdapterCols<uint8_t, 8>));
            break;
        case 3:
            assert(widths[i] == sizeof(AccessAdapterCols<uint8_t, 16>));
            break;
        case 4:
            assert(widths[i] == sizeof(AccessAdapterCols<uint8_t, 32>));
            break;
        default:
            assert(false);
        }
    }
}

__global__ void access_adapters_tracegen(
    Fp **d_trace,
    size_t num_adapters,
    size_t const *d_unpadded_heights,
    size_t const *d_widths,
    size_t num_records,
    uint8_t const *d_records,
    uint32_t *d_record_offsets,
    uint32_t *d_range_checker,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_records) {
        return;
    }
#ifdef CUDA_DEBUG
    if (idx == 0) {
        assert_widths(d_widths, num_adapters);
    }
#endif

    uint32_t record_offset = d_record_offsets[idx * (num_adapters + 1)];
    uint32_t *const row_ids = d_record_offsets + idx * (num_adapters + 1) + 1;

    d_records += record_offset;
    AccessAdapterRecordHeader const *header =
        reinterpret_cast<AccessAdapterRecordHeader const *>(d_records);
    d_records += sizeof(AccessAdapterRecordHeader);
    size_t const num_timestamps = header->block_size / header->lowest_block_size;
    uint32_t const *timestamps = reinterpret_cast<uint32_t const *>(d_records);
    d_records += sizeof(uint32_t) * num_timestamps;
    size_t const data_len = header->block_size * header->type_size;
    uint8_t const *data = d_records;

    uint32_t const timestamp = header->timestamp_and_mask & ~MERGE_AND_NOT_SPLIT_FLAG;
    size_t const log_min_block_size = __ffs(header->lowest_block_size) - 1;
    size_t const log_max_block_size = __ffs(header->block_size) - 1;
    if (header->timestamp_and_mask & MERGE_AND_NOT_SPLIT_FLAG) {
        for (size_t i = log_min_block_size; i < log_max_block_size; ++i) {
            size_t const seg_len = header->type_size << i;
            size_t const ts_len = 1 << (i - log_min_block_size);
            for (size_t j = 0; j < data_len / 2 / seg_len; ++j) {
                fill_trace_row(
                    2 << i,
                    d_trace[i],
                    row_ids[i]++,
                    false,
                    header->address_space,
                    header->pointer + j * (2 << i),
                    data + j * 2 * seg_len,
                    *std::max_element(
                        timestamps + 2 * j * ts_len, timestamps + (2 * j + 1) * ts_len
                    ),
                    *std::max_element(
                        timestamps + (2 * j + 1) * ts_len, timestamps + (2 * j + 2) * ts_len
                    ),
                    header->type_size,
                    d_unpadded_heights[i],
                    d_range_checker,
                    range_checker_bins,
                    timestamp_max_bits
                );
            }
        }
    } else {
        for (size_t i = log_min_block_size; i < log_max_block_size; ++i) {
            size_t const seg_len = header->type_size << i;
            for (size_t j = 0; j < data_len / 2 / seg_len; ++j) {
                fill_trace_row(
                    2 << i,
                    d_trace[i],
                    row_ids[i]++,
                    true,
                    header->address_space,
                    header->pointer + j * (2 << i),
                    data + j * 2 * seg_len,
                    timestamp,
                    timestamp,
                    header->type_size,
                    d_unpadded_heights[i],
                    d_range_checker,
                    range_checker_bins,
                    timestamp_max_bits
                );
            }
        }
    }
}

extern "C" int _access_adapters_tracegen(
    Fp **d_trace,
    size_t num_adapters,
    size_t const *d_unpadded_heights,
    size_t const *d_widths,
    size_t num_records,
    uint8_t const *d_records,
    uint32_t *d_record_offsets,
    uint32_t *d_range_checker,
    uint32_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    auto [grid, block] = kernel_launch_params(num_records, 512);
    access_adapters_tracegen<<<grid, block>>>(
        d_trace,
        num_adapters,
        d_unpadded_heights,
        d_widths,
        num_records,
        d_records,
        d_record_offsets,
        d_range_checker,
        range_checker_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}