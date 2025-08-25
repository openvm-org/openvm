#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/encoder.cuh"
#include "primitives/trace_access.h"
#include "system/native_adapter.cuh"

struct PublicValuesCoreRecord {
    Fp value;
    Fp index;
};

struct PublicValuesRecord {
    NativeAdapterRecord<Fp, 2, 0> adapter;
    PublicValuesCoreRecord core;
};

template <typename T> struct PublicValuesCols {
    NativeAdapterCols<T, 2, 0> adapter;
    T is_valid;
    T value;
    T index;

    // In OpenVM this is a dynamically-sized vector. We include it here
    // as a single column for use with COL_INDEX.
    T custom_pv_vars;
};

__global__ void public_values_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<PublicValuesRecord> records,
    uint32_t *range_checker,
    uint32_t range_checker_bins,
    uint32_t timestamp_max_bits,
    uint32_t num_custom_pvs,
    uint32_t max_degree,
    uint32_t k
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = NativeAdapter<Fp, 2, 0>(
            VariableRangeChecker(range_checker, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(
            row.slice_from(COL_INDEX(PublicValuesCols, adapter)), record.adapter
        );

        COL_WRITE_VALUE(row, PublicValuesCols, is_valid, Fp::one());
        COL_WRITE_VALUE(row, PublicValuesCols, value, record.core.value);
        COL_WRITE_VALUE(row, PublicValuesCols, index, record.core.index);

        auto encoder = Encoder(num_custom_pvs, max_degree, true, k);

        // We only need the starting index of custom_pv_vars to create a valid RowSlice
        encoder.write_flag_pt(
            row.slice_from(COL_INDEX(PublicValuesCols, custom_pv_vars)),
            record.core.index.asUInt32()
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _public_values_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<PublicValuesRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_bins,
    uint32_t timestamp_max_bits,
    uint32_t num_custom_pvs,
    uint32_t max_degree
) {
    assert((height & (height - 1)) == 0);
    uint32_t k = compute_k(num_custom_pvs, max_degree, true);
    assert(width == (sizeof(PublicValuesCols<uint8_t>) - 1 + k));

    auto [grid, block] = kernel_launch_params(height);
    public_values_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_bins,
        timestamp_max_bits,
        num_custom_pvs,
        max_degree,
        k
    );
    return cudaGetLastError();
}
