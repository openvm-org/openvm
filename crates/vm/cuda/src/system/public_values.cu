#include "launcher.cuh"
#include "primitives/shared_buffer.cuh"
#include "primitives/trace_access.h"

#include <cassert>
#include <cstdint>

static constexpr size_t PV_DIGEST_WIDTH = 8;
static constexpr size_t PV_LIMBS = 4;
static constexpr size_t PV_POSEIDON_WIDTH = 2 * PV_DIGEST_WIDTH;
static constexpr uint32_t PV_INIT_DOMAIN = 0x50561001;
static constexpr uint32_t PV_EVENT_DOMAIN = 0x50561002;

template <typename T> struct PublicValuesCols {
    T is_valid;
    T ordinal;
    T commit[PV_DIGEST_WIDTH];
    T hash[PV_DIGEST_WIDTH];
    T value[PV_LIMBS];
};

template <typename T> struct PublicValuesPvs {
    T initial_commit[PV_DIGEST_WIDTH];
    T final_commit[PV_DIGEST_WIDTH];
};

__device__ FpArray<PV_DIGEST_WIDTH> compress_without_record(
    FpArray<PV_DIGEST_WIDTH> const &left,
    FpArray<PV_DIGEST_WIDTH> const &right
) {
    FpArray<PV_POSEIDON_WIDTH> input;
#pragma unroll
    for (size_t i = 0; i < PV_DIGEST_WIDTH; ++i) {
        input.v[i] = left.v[i];
        input.v[PV_DIGEST_WIDTH + i] = right.v[i];
    }
    poseidon2::poseidon2_mix(reinterpret_cast<Fp *>(input.v));

    FpArray<PV_DIGEST_WIDTH> output;
#pragma unroll
    for (size_t i = 0; i < PV_DIGEST_WIDTH; ++i) {
        output.v[i] = input.v[i];
    }
    return output;
}

__device__ FpArray<PV_DIGEST_WIDTH> event_block(Fp const *value) {
    FpArray<PV_DIGEST_WIDTH> event;
    event.v[0] = Fp(PV_EVENT_DOMAIN).asRaw();
#pragma unroll
    for (size_t i = 0; i < PV_LIMBS; ++i) {
        event.v[1 + i] = value[i].asRaw();
    }
    event.v[5] = 0;
    event.v[6] = 0;
    event.v[7] = 0;
    return event;
}

__global__ void public_values_tracegen(
    Fp *trace,
    size_t height,
    Fp const *values,
    size_t initial_len,
    size_t final_len,
    Fp *pvs,
    FpArray<PV_POSEIDON_WIDTH> *poseidon2_buffer,
    uint32_t *poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    // The accumulator is intentionally serial. Public-output capacities are small, and a single
    // thread avoids intermediate prefix buffers while matching the ordered CPU transition exactly.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    FpArray<PV_DIGEST_WIDTH> commit;
#pragma unroll
    for (size_t i = 0; i < PV_DIGEST_WIDTH; ++i) {
        commit.v[i] = 0;
    }
    commit.v[0] = Fp(PV_INIT_DOMAIN).asRaw();
    commit.v[1] = Fp(static_cast<uint32_t>(height)).asRaw();

    // Reconstruct the segment's initial accumulator from the globally ordered prefix. These
    // compressions define the boundary value but are not interactions in this segment's AIR.
    for (size_t index = 0; index < initial_len; ++index) {
        Fp const *value = values + index * PV_LIMBS;
        FpArray<PV_DIGEST_WIDTH> event = event_block(value);
        commit = compress_without_record(commit, event);
    }

    PublicValuesPvs<Fp> *public_values = reinterpret_cast<PublicValuesPvs<Fp> *>(pvs);
#pragma unroll
    for (size_t i = 0; i < PV_DIGEST_WIDTH; ++i) {
        public_values->initial_commit[i] = Fp::fromRaw(commit.v[i]);
    }
    Poseidon2Buffer poseidon2(poseidon2_buffer, poseidon2_buffer_idx, poseidon2_capacity);
    size_t segment_len = final_len - initial_len;
    for (size_t ordinal = 0; ordinal < height; ++ordinal) {
        bool is_valid = ordinal < segment_len;
        size_t global_index = initial_len + (ordinal < segment_len ? ordinal : segment_len);
        Fp const *value = is_valid ? values + global_index * PV_LIMBS : nullptr;
        Fp zero_value[PV_LIMBS] = {Fp::zero(), Fp::zero(), Fp::zero(), Fp::zero()};
        if (!is_valid) {
            value = zero_value;
        }
        FpArray<PV_DIGEST_WIDTH> hash = commit;
        if (is_valid) {
            FpArray<PV_DIGEST_WIDTH> event = event_block(value);
            hash = poseidon2.compress_and_record(commit, event);
        }

        RowSlice row(trace + ordinal, height);
        COL_WRITE_VALUE(row, PublicValuesCols, is_valid, is_valid);
        COL_WRITE_VALUE(row, PublicValuesCols, ordinal, static_cast<uint32_t>(ordinal));
        COL_WRITE_ARRAY(
            row,
            PublicValuesCols,
            commit,
            reinterpret_cast<Fp const *>(commit.v)
        );
        COL_WRITE_ARRAY(
            row,
            PublicValuesCols,
            hash,
            reinterpret_cast<Fp const *>(hash.v)
        );
        COL_WRITE_ARRAY(row, PublicValuesCols, value, value);

        if (is_valid) {
            commit = hash;
        }
    }

#pragma unroll
    for (size_t i = 0; i < PV_DIGEST_WIDTH; ++i) {
        public_values->final_commit[i] = Fp::fromRaw(commit.v[i]);
    }
}

extern "C" int _public_values_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    Fp const *d_values,
    size_t initial_len,
    size_t final_len,
    Fp *d_pvs,
    Fp *d_poseidon2_raw_buffer,
    uint32_t *d_poseidon2_buffer_idx,
    size_t poseidon2_capacity,
    cudaStream_t stream
) {
    assert(height > 0 && (height & (height - 1)) == 0);
    assert(height <= UINT32_MAX);
    assert(initial_len <= final_len && final_len <= height);
    assert(width == sizeof(PublicValuesCols<uint8_t>));
    assert(poseidon2_capacity % PV_POSEIDON_WIDTH == 0);

    auto *poseidon2_buffer =
        reinterpret_cast<FpArray<PV_POSEIDON_WIDTH> *>(d_poseidon2_raw_buffer);
    size_t poseidon2_record_capacity = poseidon2_capacity / PV_POSEIDON_WIDTH;
    public_values_tracegen<<<1, 1, 0, stream>>>(
        d_trace,
        height,
        d_values,
        initial_len,
        final_len,
        d_pvs,
        poseidon2_buffer,
        d_poseidon2_buffer_idx,
        poseidon2_record_capacity
    );
    return CHECK_KERNEL();
}
