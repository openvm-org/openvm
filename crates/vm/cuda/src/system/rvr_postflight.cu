#define CUB_WRAPPED_NAMESPACE openvm_rvr_postflight_cub
#include "arch/rvr/replay.cuh"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "system/memory/touched_block.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cstddef>
#include <cstdint>

namespace cub = openvm_rvr_postflight_cub::cub;

namespace {

static constexpr uint32_t ERROR_MEMORY_TIMESTAMPS = 101;
static constexpr uint32_t ERROR_SORT_ORDER = 103;
static constexpr uint32_t ERROR_MEMORY_ADDRESS = 107;
static constexpr uint32_t ERROR_MEMORY_MASK = 108;
static constexpr uint32_t ERROR_INITIAL_MEMORY = 109;
static constexpr uint32_t ERROR_MEMORY_CHRONOLOGY = 110;
static constexpr uint32_t ERROR_FIELD_VALUE = 118;
static constexpr uint32_t ERROR_FIELD_REFERENCE = 119;

static constexpr uint32_t MEMORY_CELL_U16 = 1;
static constexpr uint32_t MEMORY_CELL_FIELD32 = 2;
static constexpr uint8_t FIELD_FULL_WRITE_MASK = 0xff;
static constexpr int BLOCK_KEY_BEGIN_BIT = 32;
static constexpr int BLOCK_KEY_END_BIT = 64;

struct RvrMemoryAddressSpace {
    uint64_t num_cells;
    uint32_t cell_kind;
    uint32_t padding;
};

static_assert(sizeof(RvrMemoryAddressSpace) == 16);
static_assert(offsetof(PreflightMemoryEvent, value) % alignof(uint32_t) == 0);
using RvrTouchedBlock = MemoryTouchedBlock;
using AliasedU32 = uint32_t __attribute__((may_alias));

struct ValueChunk {
    uint64_t bytes;
    uint32_t valid;
    uint32_t padding;
};

static_assert(sizeof(ValueChunk) == 2 * sizeof(uint64_t));
static constexpr uint32_t VALUE_CHUNK_DIRTY = 1u << 31;
static constexpr uint32_t VALUE_CHUNK_VALID = 0xff;

struct BlockKeyEqual {
    __host__ __device__ __forceinline__ bool operator()(uint64_t lhs, uint64_t rhs) const {
        return uint32_t(lhs >> 32) == uint32_t(rhs >> 32);
    }
};

struct LastWriteWins {
    __host__ __device__ __forceinline__ ValueChunk
    operator()(ValueChunk prefix, ValueChunk current) const {
        ValueChunk result = prefix;
        for (uint32_t lane = 0; lane < 8; ++lane) {
            uint32_t bit = 1u << lane;
            if ((current.valid & bit) != 0) {
                uint32_t shift = 8 * lane;
                uint64_t mask = uint64_t{0xff} << shift;
                result.bytes = (result.bytes & ~mask) | (current.bytes & mask);
            }
        }
        result.valid |= current.valid;
        return result;
    }
};

__device__ bool compact_block_key(
    uint32_t address_space,
    uint32_t pointer,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    uint32_t address_space_offset,
    uint32_t address_space_height,
    uint32_t pointer_max_bits,
    bool allow_field,
    uint32_t &out
) {
    // Pointers here count AS-native cells (u16 or field32 per `cell_kind`), so the
    // per-AS `num_cells` check below is the authoritative pointer-domain bound;
    // `pointer_limit` only guarantees the packed key fits `block_pointer_bits`.
    uint64_t address_space_limit =
        static_cast<uint64_t>(address_space_offset) + (uint64_t{1} << address_space_height);
    uint64_t pointer_limit = uint64_t{1} << pointer_max_bits;
    if (address_space < address_space_offset || address_space >= address_space_limit ||
        address_space >= address_spaces.len() || pointer >= pointer_limit || pointer % 4 != 0) {
        return false;
    }
    auto const &config = address_spaces[address_space];
    if ((config.cell_kind != MEMORY_CELL_U16 &&
         !(allow_field && config.cell_kind == MEMORY_CELL_FIELD32)) ||
        static_cast<uint64_t>(pointer) + 4 > config.num_cells) {
        return false;
    }
    uint32_t block_pointer_bits = pointer_max_bits - 2;
    out = ((address_space - address_space_offset) << block_pointer_bits) | (pointer >> 2);
    return true;
}

__device__ bool initial_quad(
    uint32_t address_space,
    uint32_t pointer,
    uint32_t byte_offset,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    DeviceBufferConstView<DeviceRawBufferConstView> initial_memory,
    uint8_t (&out)[4]
) {
    if (address_space >= address_spaces.len() || address_space >= initial_memory.len()) {
        return false;
    }
    auto const &config = address_spaces[address_space];
    uint32_t cell_bytes;
    if (config.cell_kind == MEMORY_CELL_U16) {
        cell_bytes = 2;
    } else if (config.cell_kind == MEMORY_CELL_FIELD32) {
        cell_bytes = 4;
    } else {
        return false;
    }
    if (byte_offset + 4 > 4 * cell_bytes) return false;
    auto image = initial_memory[address_space].as_typed<uint8_t>();
    uint64_t byte_pointer = uint64_t(pointer) * cell_bytes + byte_offset;
    if (byte_pointer + 4 > image.len()) return false;
#pragma unroll
    for (uint32_t lane = 0; lane < 4; ++lane) out[lane] = image[byte_pointer + lane];
    return true;
}

__device__ __forceinline__ uint32_t field_reference(PreflightMemoryEvent const &event) {
    return uint32_t(event.value[0]) | (uint32_t(event.value[1]) << 16);
}

__device__ __forceinline__ void set_field_reference(uint16_t (&value)[4], uint32_t reference) {
    value[0] = uint16_t(reference);
    value[1] = uint16_t(reference >> 16);
    value[2] = 0;
    value[3] = 0;
}

__device__ __forceinline__ bool field_block_is_valid(RvrFieldBlock const &block) {
#pragma unroll
    for (uint32_t lane = 0; lane < 4; ++lane) {
        if (block.values[lane] >= Fp::P) return false;
    }
    return true;
}

__device__ bool load_initial_field_block(
    uint32_t address_space,
    uint32_t pointer,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    DeviceBufferConstView<DeviceRawBufferConstView> initial_memory,
    RvrFieldBlock &out
) {
#pragma unroll
    for (uint32_t quad = 0; quad < 4; ++quad) {
        uint8_t bytes[4];
        if (!initial_quad(
                address_space, pointer, 4 * quad, address_spaces, initial_memory, bytes
            )) {
            return false;
        }
        out.values[quad] = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
                           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }
    return field_block_is_valid(out);
}

__global__ void prepare_chronology_entries(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint8_t> write_masks,
    DeviceBufferConstView<RvrFieldBlock> field_values,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    uint32_t address_space_offset,
    uint32_t address_space_height,
    uint32_t pointer_max_bits,
    uint32_t field_address_space,
    uint64_t *keys,
    uint32_t *error
) {
    size_t ordinal = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (ordinal >= memory.len()) return;
    auto const &event = memory[ordinal];
    uint8_t mask = write_masks[ordinal];
    uint32_t compact_key;
    bool address_valid = compact_block_key(
            preflight_address_space(event),
            event.pointer,
            address_spaces,
            address_space_offset,
            address_space_height,
            pointer_max_bits,
            true,
            compact_key
        );
    if (!address_valid) {
        preflight_set_error(error, ERROR_MEMORY_ADDRESS);
        compact_key = 0;
    }
    bool is_write = preflight_is_write(event);
    if (is_write != (mask != 0)) preflight_set_error(error, ERROR_MEMORY_MASK);
    uint32_t cell_kind = address_valid
                             ? address_spaces[preflight_address_space(event)].cell_kind
                             : MEMORY_CELL_U16;
    if (cell_kind == MEMORY_CELL_FIELD32) {
        if (preflight_address_space(event) != field_address_space) {
            preflight_set_error(error, ERROR_MEMORY_ADDRESS);
        }
        uint32_t reference = field_reference(event);
        if (event.value[2] != 0 || event.value[3] != 0 || reference >= field_values.len()) {
            preflight_set_error(error, ERROR_FIELD_REFERENCE);
        } else if ((is_write && mask != FIELD_FULL_WRITE_MASK) ||
                   (!is_write && mask != 0)) {
            preflight_set_error(error, ERROR_MEMORY_MASK);
        } else if (is_write && !field_block_is_valid(field_values[reference])) {
            preflight_set_error(error, ERROR_FIELD_VALUE);
        } else if (!is_write) {
#pragma unroll
            for (uint32_t lane = 0; lane < 4; ++lane) {
                if (field_values[reference].values[lane] != 0) {
                    preflight_set_error(error, ERROR_FIELD_VALUE);
                }
            }
        }
    } else {
        auto const *patch = reinterpret_cast<uint8_t const *>(event.value);
#pragma unroll
        for (uint32_t lane = 0; lane < 8; ++lane) {
            if ((mask & (1u << lane)) == 0 && patch[lane] != 0) {
                preflight_set_error(error, ERROR_MEMORY_MASK);
            }
        }
    }
    keys[ordinal] = (uint64_t(compact_key) << 32) | uint32_t(ordinal);
    if (ordinal != 0 && memory[ordinal - 1].timestamp >= event.timestamp) {
        preflight_set_error(error, ERROR_MEMORY_TIMESTAMPS);
    }
}

__global__ void mark_chronology_metadata(
    DeviceBufferConstView<uint8_t> write_masks,
    uint64_t const *sorted_keys,
    size_t num_entries,
    uint64_t *packed_flags
) {
    size_t pos = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (pos >= num_entries) return;
    uint32_t key = uint32_t(sorted_keys[pos] >> 32);
    bool head = pos == 0 || uint32_t(sorted_keys[pos - 1] >> 32) != key;
    bool tail = pos + 1 == num_entries || uint32_t(sorted_keys[pos + 1] >> 32) != key;
    uint32_t ordinal = uint32_t(sorted_keys[pos]);
    uint64_t seed = uint64_t(head && write_masks[ordinal] != 0);
    uint64_t touched = uint64_t(tail) << 32;
    packed_flags[pos] = seed | touched;
}

__device__ size_t chronology_key_lower_bound(
    uint64_t const *sorted_keys,
    size_t num_entries,
    uint64_t target
) {
    size_t left = 0;
    size_t right = num_entries;
    while (left < right) {
        size_t middle = left + (right - left) / 2;
        if ((sorted_keys[middle] >> 32) < target) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__device__ uint32_t chronology_seed_prefix(
    DeviceBufferConstView<uint8_t> write_masks,
    uint64_t const *sorted_keys,
    uint64_t const *packed_positions,
    size_t end
) {
    if (end == 0) return 0;
    size_t last = end - 1;
    uint32_t ordinal = uint32_t(sorted_keys[last]);
    bool head = last == 0 ||
                uint32_t(sorted_keys[last - 1] >> 32) !=
                    uint32_t(sorted_keys[last] >> 32);
    return uint32_t(packed_positions[last]) +
           uint32_t(head && write_masks[ordinal] != 0);
}

__global__ void finish_chronology_counts(
    DeviceBufferConstView<uint8_t> write_masks,
    uint64_t const *sorted_keys,
    uint64_t const *packed_positions,
    size_t num_entries,
    uint32_t address_space_offset,
    uint32_t pointer_max_bits,
    uint32_t field_address_space,
    bool count_field_metadata,
    uint32_t *counts
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (num_entries == 0) {
        counts[0] = 0;
        counts[1] = 0;
        if (count_field_metadata) {
            counts[2] = 0;
            counts[3] = 0;
            counts[4] = 0;
            counts[5] = 0;
        }
        return;
    }
    size_t pos = num_entries - 1;
    uint32_t key = uint32_t(sorted_keys[pos] >> 32);
    bool head = pos == 0 || uint32_t(sorted_keys[pos - 1] >> 32) != key;
    uint32_t ordinal = uint32_t(sorted_keys[pos]);
    uint64_t total = packed_positions[pos] + uint64_t(head && write_masks[ordinal] != 0) +
                     (uint64_t{1} << 32);
    counts[0] = uint32_t(total);
    counts[1] = uint32_t(total >> 32);
    if (!count_field_metadata) return;

    uint32_t block_pointer_bits = pointer_max_bits - 2;
    uint64_t field_key_begin =
        uint64_t(field_address_space - address_space_offset) << block_pointer_bits;
    uint64_t field_key_end =
        uint64_t(field_address_space - address_space_offset + 1) << block_pointer_bits;
    size_t field_begin =
        chronology_key_lower_bound(sorted_keys, num_entries, field_key_begin);
    size_t field_end =
        chronology_key_lower_bound(sorted_keys, num_entries, field_key_end);
    uint32_t field_seed_begin =
        chronology_seed_prefix(write_masks, sorted_keys, packed_positions, field_begin);
    uint32_t field_seed_end =
        chronology_seed_prefix(write_masks, sorted_keys, packed_positions, field_end);
    counts[2] = uint32_t(field_begin);
    counts[3] = uint32_t(field_end);
    counts[4] = field_seed_begin;
    counts[5] = field_seed_end - field_seed_begin;
}

__global__ void scatter_chronology_metadata(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint8_t> write_masks,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    DeviceBufferConstView<DeviceRawBufferConstView> initial_memory,
    uint64_t const *sorted_keys,
    uint64_t const *packed_positions,
    size_t num_entries,
    uint32_t *predecessors,
    PreflightInitialWrite *seeds,
    size_t num_seeds,
    RvrFieldBlock *field_seeds,
    size_t num_field_seeds,
    uint32_t field_seed_base,
    RvrTouchedBlock *touched,
    size_t num_touched,
    uint32_t *error
) {
    size_t pos = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (pos >= num_entries) return;
    uint32_t key = uint32_t(sorted_keys[pos] >> 32);
    uint32_t ordinal = uint32_t(sorted_keys[pos]);
    bool head = pos == 0 || uint32_t(sorted_keys[pos - 1] >> 32) != key;
    bool tail = pos + 1 == num_entries || uint32_t(sorted_keys[pos + 1] >> 32) != key;
    auto const &event = memory[ordinal];
    uint64_t positions = packed_positions[pos];

    if (head) {
        if (write_masks[ordinal] == 0) {
            predecessors[ordinal] = 0;
        } else {
            uint32_t seed_index = uint32_t(positions);
            if (seed_index >= num_seeds) {
                preflight_set_error(error, ERROR_MEMORY_CHRONOLOGY);
                return;
            }
            auto &seed = seeds[seed_index];
            seed.address_space = preflight_address_space(event);
            seed.pointer = event.pointer;
            auto const &config = address_spaces[preflight_address_space(event)];
            if (config.cell_kind == MEMORY_CELL_FIELD32) {
                if (seed_index < field_seed_base ||
                    seed_index - field_seed_base >= num_field_seeds) {
                    preflight_set_error(error, ERROR_MEMORY_CHRONOLOGY);
                    return;
                }
                uint32_t field_seed_index = seed_index - field_seed_base;
                RvrFieldBlock initial;
                if (!load_initial_field_block(
                        preflight_address_space(event),
                        event.pointer,
                        address_spaces,
                        initial_memory,
                        initial
                    )) {
                    preflight_set_error(error, ERROR_INITIAL_MEMORY);
                    return;
                }
                field_seeds[field_seed_index] = initial;
                set_field_reference(seed.initial_value, field_seed_index);
            } else {
                uint8_t initial[8];
#pragma unroll
                for (uint32_t quad = 0; quad < 2; ++quad) {
                    uint8_t bytes[4];
                    if (!initial_quad(
                            preflight_address_space(event),
                            event.pointer,
                            4 * quad,
                            address_spaces,
                            initial_memory,
                            bytes
                        )) {
                        preflight_set_error(error, ERROR_INITIAL_MEMORY);
                        return;
                    }
#pragma unroll
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        initial[4 * quad + lane] = bytes[lane];
                    }
                }
#pragma unroll
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    seed.initial_value[lane] = uint16_t(initial[2 * lane]) |
                                               (uint16_t(initial[2 * lane + 1]) << 8);
                }
            }
            predecessors[ordinal] = MEMORY_PREDECESSOR_SEED_BIT | seed_index;
        }
    } else {
        uint32_t previous_ordinal = uint32_t(sorted_keys[pos - 1]);
        if (previous_ordinal >= ordinal) {
            preflight_set_error(error, ERROR_SORT_ORDER);
            return;
        }
        predecessors[ordinal] = previous_ordinal + 1;
    }

    if (tail) {
        uint32_t touched_index = uint32_t(positions >> 32);
        if (touched_index >= num_touched) {
            preflight_set_error(error, ERROR_MEMORY_CHRONOLOGY);
            return;
        }
        auto &record = touched[touched_index];
        record.address_space = preflight_address_space(event);
        record.ptr = event.pointer;
        record.is_dirty = 0;
        record.timestamp = event.timestamp;
        record.values[0] = static_cast<uint32_t>(pos);
        record.values[1] = 0;
        record.values[2] = 0;
        record.values[3] = 0;
    }
}

__global__ void prepare_value_chunks(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint8_t> write_masks,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    DeviceBufferConstView<DeviceRawBufferConstView> initial_memory,
    DeviceBufferConstView<RvrFieldBlock> field_values,
    uint64_t const *sorted_keys,
    size_t sorted_offset,
    size_t num_entries,
    uint32_t byte_offset,
    ValueChunk *chunks,
    uint32_t *error
) {
    size_t pos = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (pos >= num_entries) return;
    size_t sorted_pos = sorted_offset + pos;
    uint32_t key = uint32_t(sorted_keys[sorted_pos] >> 32);
    uint32_t ordinal = uint32_t(sorted_keys[sorted_pos]);
    bool head = sorted_pos == 0 || uint32_t(sorted_keys[sorted_pos - 1] >> 32) != key;
    auto const &event = memory[ordinal];
    auto const &config = address_spaces[preflight_address_space(event)];
    ValueChunk chunk{0, 0, 0};
    if (head) {
#pragma unroll
        for (uint32_t quad = 0; quad < 2; ++quad) {
            uint8_t initial[4];
            if (!initial_quad(
                    preflight_address_space(event),
                    event.pointer,
                    byte_offset + 4 * quad,
                    address_spaces,
                    initial_memory,
                    initial
                )) {
                preflight_set_error(error, ERROR_INITIAL_MEMORY);
                chunks[sorted_pos] = chunk;
                return;
            }
#pragma unroll
            for (uint32_t lane = 0; lane < 4; ++lane) {
                chunk.bytes |= uint64_t(initial[lane]) << (8 * (4 * quad + lane));
            }
        }
        chunk.valid = VALUE_CHUNK_VALID;
    }
    uint8_t mask;
    uint8_t const *patch;
    if (config.cell_kind == MEMORY_CELL_FIELD32) {
        uint32_t reference = field_reference(event);
        if (reference >= field_values.len()) {
            preflight_set_error(error, ERROR_FIELD_REFERENCE);
            chunks[sorted_pos] = chunk;
            return;
        }
        mask = write_masks[ordinal] == FIELD_FULL_WRITE_MASK ? VALUE_CHUNK_VALID : 0;
        patch = reinterpret_cast<uint8_t const *>(&field_values[reference]);
    } else {
        mask = write_masks[ordinal];
        patch = reinterpret_cast<uint8_t const *>(event.value);
    }
#pragma unroll
    for (uint32_t lane = 0; lane < 8; ++lane) {
        uint32_t bit = 1u << lane;
        if ((mask & bit) != 0) {
            uint32_t shift = 8 * lane;
            uint64_t lane_mask = uint64_t{0xff} << shift;
            chunk.bytes = (chunk.bytes & ~lane_mask) |
                          (uint64_t(patch[byte_offset + lane]) << shift);
            chunk.valid |= bit;
        }
    }
    if (write_masks[ordinal] != 0) {
        chunk.valid |= VALUE_CHUNK_DIRTY;
    }
    chunks[sorted_pos] = chunk;
}

__global__ void scatter_value_chunks(
    DeviceBufferView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    DeviceBufferView<RvrFieldBlock> field_values,
    uint64_t const *sorted_keys,
    size_t sorted_offset,
    size_t num_entries,
    uint32_t byte_offset,
    ValueChunk const *chunks,
    uint32_t *error
) {
    size_t pos = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (pos >= num_entries) return;
    size_t sorted_pos = sorted_offset + pos;
    auto const &chunk = chunks[sorted_pos];
    if ((chunk.valid & VALUE_CHUNK_VALID) != VALUE_CHUNK_VALID) {
        preflight_set_error(error, ERROR_MEMORY_CHRONOLOGY);
        return;
    }
    uint32_t ordinal = uint32_t(sorted_keys[sorted_pos]);
    auto const &event = memory[ordinal];
    auto const &config = address_spaces[preflight_address_space(event)];
    if (config.cell_kind == MEMORY_CELL_FIELD32) {
        uint32_t reference = field_reference(event);
        if (reference >= field_values.len()) {
            preflight_set_error(error, ERROR_FIELD_REFERENCE);
            return;
        }
        auto *words = field_values[reference].values;
        uint32_t word_offset = byte_offset / sizeof(uint32_t);
        words[word_offset] = uint32_t(chunk.bytes);
        words[word_offset + 1] = uint32_t(chunk.bytes >> 32);
    } else {
        // The transcript ABI aligns this fixed eight-byte payload to one
        // word. Store it as two words so replay resolution does not expand
        // back into eight independent byte stores.
        auto *words = reinterpret_cast<AliasedU32 *>(memory[ordinal].value);
        words[0] = uint32_t(chunk.bytes);
        words[1] = uint32_t(chunk.bytes >> 32);
    }
    if (config.cell_kind == MEMORY_CELL_FIELD32) {
        uint32_t reference = field_reference(event);
        if (!field_block_is_valid(field_values[reference])) {
            preflight_set_error(error, ERROR_FIELD_VALUE);
        }
    }
}

__global__ void finalize_chronology_touched(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    DeviceBufferConstView<RvrFieldBlock> field_values,
    uint64_t const *sorted_keys,
    ValueChunk const *chunks,
    RvrTouchedBlock *touched,
    size_t num_touched,
    uint32_t *error
) {
    size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (index >= num_touched) return;
    auto &record = touched[index];
    uint32_t sorted_pos = record.values[0];
    if (sorted_pos >= memory.len()) {
        preflight_set_error(error, ERROR_MEMORY_CHRONOLOGY);
        return;
    }
    uint32_t key = uint32_t(sorted_keys[sorted_pos] >> 32);
    if (sorted_pos + 1 < memory.len() &&
        uint32_t(sorted_keys[sorted_pos + 1] >> 32) == key) {
        preflight_set_error(error, ERROR_MEMORY_CHRONOLOGY);
        return;
    }
    uint32_t ordinal = uint32_t(sorted_keys[sorted_pos]);
    if (ordinal >= memory.len()) {
        preflight_set_error(error, ERROR_MEMORY_CHRONOLOGY);
        return;
    }
    record.is_dirty = uint32_t((chunks[sorted_pos].valid & VALUE_CHUNK_DIRTY) != 0);
    auto const &event = memory[ordinal];
    auto const &config = address_spaces[preflight_address_space(event)];
    if (config.cell_kind == MEMORY_CELL_FIELD32) {
        uint32_t reference = field_reference(event);
        if (reference >= field_values.len() || !field_block_is_valid(field_values[reference])) {
            preflight_set_error(error, ERROR_FIELD_VALUE);
            return;
        }
#pragma unroll
        for (uint32_t lane = 0; lane < 4; ++lane) {
            record.values[lane] = field_values[reference].values[lane];
        }
    } else {
#pragma unroll
        for (uint32_t lane = 0; lane < 4; ++lane) {
            record.values[lane] = Fp(event.value[lane]).asRaw();
        }
    }
}

} // namespace

extern "C" int _rvr_memory_chronology_get_temp_bytes(
    size_t num_entries,
    size_t *h_temp_bytes_out,
    cudaStream_t stream
) {
    size_t sort_temp_bytes = 0;
    size_t count_temp_bytes = 0;
    size_t chronology_temp_bytes = 0;
    if (num_entries != 0) {
        cub::DeviceRadixSort::SortKeys(
            nullptr,
            sort_temp_bytes,
            static_cast<uint64_t *>(nullptr),
            static_cast<uint64_t *>(nullptr),
            num_entries,
            BLOCK_KEY_BEGIN_BIT,
            BLOCK_KEY_END_BIT,
            stream
        );
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            count_temp_bytes,
            static_cast<uint64_t *>(nullptr),
            static_cast<uint64_t *>(nullptr),
            num_entries,
            stream
        );
        cub::DeviceScan::InclusiveScanByKey(
            nullptr,
            chronology_temp_bytes,
            static_cast<uint64_t *>(nullptr),
            static_cast<ValueChunk *>(nullptr),
            static_cast<ValueChunk *>(nullptr),
            LastWriteWins{},
            num_entries,
            BlockKeyEqual{},
            stream
        );
    }
    size_t temp_bytes = sort_temp_bytes > count_temp_bytes ? sort_temp_bytes : count_temp_bytes;
    *h_temp_bytes_out = temp_bytes > chronology_temp_bytes ? temp_bytes : chronology_temp_bytes;
    return CHECK_KERNEL();
}

extern "C" int _rvr_memory_chronology_sort_and_count(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint8_t> write_masks,
    DeviceBufferConstView<RvrFieldBlock> field_values,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    uint32_t address_space_offset,
    uint32_t address_space_height,
    uint32_t pointer_max_bits,
    uint32_t field_address_space,
    uint32_t count_field_metadata,
    uint64_t *workspace,
    uint64_t *sorted_keys,
    uint32_t *counts,
    void *temp_storage,
    size_t temp_storage_bytes,
    uint32_t *error,
    cudaStream_t stream
) {
    if (memory.len() != write_masks.len()) return int(cudaErrorInvalidValue);
    uint64_t address_space_limit =
        uint64_t(address_space_offset) + (uint64_t{1} << address_space_height);
    bool has_field_metadata = count_field_metadata != 0;
    if (pointer_max_bits < 2 ||
        (has_field_metadata &&
         (field_address_space < address_space_offset ||
          field_address_space >= address_space_limit))) {
        return int(cudaErrorInvalidValue);
    }
    size_t num_entries = memory.len();
    if (num_entries == 0) {
        size_t count_bytes = count_field_metadata != 0 ? 6 : 2;
        if (cudaError_t err = cudaMemsetAsync(counts, 0, count_bytes * sizeof(uint32_t), stream);
            err != cudaSuccess) {
            return err;
        }
        return 0;
    }

    auto [grid, block] = kernel_launch_params(num_entries);
    prepare_chronology_entries<<<grid, block, 0, stream>>>(
        memory,
        write_masks,
        field_values,
        address_spaces,
        address_space_offset,
        address_space_height,
        pointer_max_bits,
        field_address_space,
        workspace,
        error
    );
    if (int err = CHECK_KERNEL(); err) return err;
    if (cudaError_t err = cub::DeviceRadixSort::SortKeys(
            temp_storage,
            temp_storage_bytes,
            workspace,
            sorted_keys,
            num_entries,
            // The low half is the input ordinal. The memory log is already in
            // strict timestamp order, and CUB's radix sort is stable, so
            // sorting only the block-key half preserves chronology within
            // each block without spending radix passes on an already ordered
            // tiebreaker.
            BLOCK_KEY_BEGIN_BIT,
            BLOCK_KEY_END_BIT,
            stream
        );
        err != cudaSuccess) {
        return err;
    }
    mark_chronology_metadata<<<grid, block, 0, stream>>>(
        write_masks, sorted_keys, num_entries, workspace
    );
    if (int err = CHECK_KERNEL(); err) return err;
    if (cudaError_t err = cub::DeviceScan::ExclusiveSum(
            temp_storage,
            temp_storage_bytes,
            workspace,
            workspace,
            num_entries,
            stream
        );
        err != cudaSuccess) {
        return err;
    }
    finish_chronology_counts<<<1, 1, 0, stream>>>(
        write_masks,
        sorted_keys,
        workspace,
        num_entries,
        address_space_offset,
        pointer_max_bits,
        field_address_space,
        has_field_metadata,
        counts
    );
    return CHECK_KERNEL();
}

extern "C" int _rvr_memory_chronology_resolve(
    DeviceBufferView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint8_t> write_masks,
    DeviceBufferConstView<RvrMemoryAddressSpace> address_spaces,
    DeviceBufferConstView<DeviceRawBufferConstView> initial_memory,
    DeviceBufferView<RvrFieldBlock> field_values,
    uint64_t const *sorted_keys,
    uint64_t *workspace,
    uint32_t *predecessors,
    DeviceBufferView<PreflightInitialWrite> seeds,
    DeviceBufferView<RvrFieldBlock> field_seeds,
    uint32_t field_begin,
    uint32_t field_end,
    uint32_t field_seed_base,
    DeviceBufferView<RvrTouchedBlock> touched,
    void *temp_storage,
    size_t temp_storage_bytes,
    uint32_t *error,
    cudaStream_t stream
) {
    if (memory.len() != write_masks.len()) return int(cudaErrorInvalidValue);
    size_t num_entries = memory.len();
    if (num_entries == 0) return 0;
    auto [grid, block] = kernel_launch_params(num_entries);
    scatter_chronology_metadata<<<grid, block, 0, stream>>>(
        DeviceBufferConstView<PreflightMemoryEvent>{memory.ptr, memory.size},
        write_masks,
        address_spaces,
        initial_memory,
        sorted_keys,
        workspace,
        num_entries,
        predecessors,
        seeds.ptr,
        seeds.len(),
        field_seeds.ptr,
        field_seeds.len(),
        field_seed_base,
        touched.ptr,
        touched.len(),
        error
    );
    if (int err = CHECK_KERNEL(); err) return err;

    auto *chunks = reinterpret_cast<ValueChunk *>(workspace);
    prepare_value_chunks<<<grid, block, 0, stream>>>(
        DeviceBufferConstView<PreflightMemoryEvent>{memory.ptr, memory.size},
        write_masks,
        address_spaces,
        initial_memory,
        DeviceBufferConstView<RvrFieldBlock>{field_values.ptr, field_values.size},
        sorted_keys,
        0,
        num_entries,
        0,
        chunks,
        error
    );
    if (int err = CHECK_KERNEL(); err) return err;
    if (cudaError_t err = cub::DeviceScan::InclusiveScanByKey(
            temp_storage,
            temp_storage_bytes,
            sorted_keys,
            chunks,
            chunks,
            LastWriteWins{},
            num_entries,
            BlockKeyEqual{},
            stream
        );
        err != cudaSuccess) {
        return err;
    }
    scatter_value_chunks<<<grid, block, 0, stream>>>(
        memory,
        address_spaces,
        field_values,
        sorted_keys,
        0,
        num_entries,
        0,
        chunks,
        error
    );
    if (int err = CHECK_KERNEL(); err) return err;

    if (field_end < field_begin || field_end > num_entries ||
        size_t(field_end - field_begin) != field_values.len()) {
        return int(cudaErrorInvalidValue);
    }
    size_t num_field_entries = field_end - field_begin;
    if (num_field_entries != 0) {
        auto [field_grid, field_block] = kernel_launch_params(num_field_entries);
        prepare_value_chunks<<<field_grid, field_block, 0, stream>>>(
            DeviceBufferConstView<PreflightMemoryEvent>{memory.ptr, memory.size},
            write_masks,
            address_spaces,
            initial_memory,
            DeviceBufferConstView<RvrFieldBlock>{field_values.ptr, field_values.size},
            sorted_keys,
            field_begin,
            num_field_entries,
            8,
            chunks,
            error
        );
        if (int err = CHECK_KERNEL(); err) return err;
        if (cudaError_t err = cub::DeviceScan::InclusiveScanByKey(
                temp_storage,
                temp_storage_bytes,
                sorted_keys + field_begin,
                chunks + field_begin,
                chunks + field_begin,
                LastWriteWins{},
                num_field_entries,
                BlockKeyEqual{},
                stream
            );
            err != cudaSuccess) {
            return err;
        }
        scatter_value_chunks<<<field_grid, field_block, 0, stream>>>(
            memory,
            address_spaces,
            field_values,
            sorted_keys,
            field_begin,
            num_field_entries,
            8,
            chunks,
            error
        );
        if (int err = CHECK_KERNEL(); err) return err;
    }

    if (touched.len() != 0) {
        auto [touched_grid, touched_block] = kernel_launch_params(touched.len());
        finalize_chronology_touched<<<touched_grid, touched_block, 0, stream>>>(
            DeviceBufferConstView<PreflightMemoryEvent>{memory.ptr, memory.size},
            address_spaces,
            DeviceBufferConstView<RvrFieldBlock>{field_values.ptr, field_values.size},
            sorted_keys,
            chunks,
            touched.ptr,
            touched.len(),
            error
        );
    }
    return CHECK_KERNEL();
}

namespace {

static constexpr uint32_t ERROR_PROGRAM_START = 111;
static constexpr uint32_t ERROR_PROGRAM_PC = 112;
static constexpr uint32_t ERROR_PROGRAM_TIMESTAMP = 113;
static constexpr uint32_t ERROR_TERMINATE_SCHEDULE = 114;
static constexpr uint32_t ERROR_ENDPOINT = 115;
static constexpr uint32_t ERROR_MEMORY_BOUNDARY = 116;
static constexpr uint32_t ERROR_TIMESTAMP_DOMAIN = 117;

struct RvrOpcodeRange {
    uint32_t start;
    uint32_t end;
};

static_assert(sizeof(RvrOpcodeRange) == 8);

__device__ size_t memory_lower_bound(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t timestamp
) {
    size_t left = 0;
    size_t right = memory.len();
    while (left < right) {
        size_t middle = left + (right - left) / 2;
        if (memory[middle].timestamp < timestamp) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__global__ void prepare_program_steps(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    DeviceBufferConstView<uint32_t> dense_program_rows,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t timestamp_max_bits,
    uint32_t endpoint_kind,
    uint32_t resume_pc,
    uint32_t final_timestamp,
    uint32_t terminate_opcode,
    uint32_t *opcode_keys,
    RvrReplayStep *steps,
    uint32_t *program_frequencies,
    uint32_t *error
) {
    size_t program_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t num_steps = program.len() - 1;
    if (program_index >= num_steps) return;

    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (program_index == 0 && from.timestamp != 1) {
        preflight_set_error(error, ERROR_PROGRAM_START);
    }
    RvrReplayInstruction const *instruction = nullptr;
    size_t instruction_slot = 0;
    instruction =
        resolve_replay_instruction(instructions, pc_base, from.pc, &instruction_slot);
    if (instruction == nullptr || instruction_slot >= dense_program_rows.len() ||
        dense_program_rows[instruction_slot] == UINT32_MAX) {
        preflight_set_error(error, ERROR_PROGRAM_PC);
        return;
    }
    uint32_t opcode = instruction->words[0];
    bool is_terminate = opcode == terminate_opcode;
    if (to.timestamp < from.timestamp || (!is_terminate && to.timestamp == from.timestamp)) {
        preflight_set_error(error, ERROR_PROGRAM_TIMESTAMP);
        return;
    }
    if (is_terminate &&
        (endpoint_kind != 0 || program_index + 1 != num_steps || from.pc != to.pc ||
         from.timestamp != to.timestamp)) {
        preflight_set_error(error, ERROR_TERMINATE_SCHEDULE);
        return;
    }

    size_t memory_start = memory_lower_bound(memory, from.timestamp);
    if (program_index == 0 && memory_start != 0) {
        preflight_set_error(error, ERROR_MEMORY_BOUNDARY);
        return;
    }

    if (program_index + 1 == num_steps) {
        if (memory_lower_bound(memory, to.timestamp) != memory.len()) {
            preflight_set_error(error, ERROR_MEMORY_BOUNDARY);
            return;
        }
        if (to.timestamp >= (uint32_t{1} << timestamp_max_bits)) {
            preflight_set_error(error, ERROR_TIMESTAMP_DOMAIN);
            return;
        }
        if (endpoint_kind == 0) {
            if (!is_terminate) {
                preflight_set_error(error, ERROR_ENDPOINT);
                return;
            }
        } else {
            RvrReplayInstruction const *resume_instruction = nullptr;
            size_t resume_slot = 0;
            resume_instruction =
                resolve_replay_instruction(instructions, pc_base, to.pc, &resume_slot);
            if (to.pc != resume_pc || to.timestamp != final_timestamp || is_terminate ||
                resume_instruction == nullptr) {
                preflight_set_error(error, ERROR_ENDPOINT);
                return;
            }
        }
    }

    opcode_keys[program_index] = opcode;
    steps[program_index] = RvrReplayStep{
        .program_index = static_cast<uint32_t>(program_index),
        .memory_start = static_cast<uint32_t>(memory_start),
    };
    atomicAdd(&program_frequencies[dense_program_rows[instruction_slot]], 1);
}

__global__ void validate_empty_program(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t timestamp_max_bits,
    uint32_t endpoint_kind,
    uint32_t resume_pc,
    uint32_t final_timestamp,
    uint32_t *error
) {
    auto const &sentinel = program[0];
    RvrReplayInstruction const *resume_instruction = nullptr;
    size_t resume_slot = 0;
    resume_instruction =
        resolve_replay_instruction(instructions, pc_base, sentinel.pc, &resume_slot);
    if (sentinel.timestamp >= (uint32_t{1} << timestamp_max_bits)) {
        preflight_set_error(error, ERROR_TIMESTAMP_DOMAIN);
    } else if (endpoint_kind != 1 || sentinel.timestamp != 1 || sentinel.pc != resume_pc ||
        sentinel.timestamp != final_timestamp || memory.len() != 0 ||
        resume_instruction == nullptr) {
        preflight_set_error(error, ERROR_ENDPOINT);
    }
}

__device__ size_t opcode_lower_bound(uint32_t const *keys, size_t len, uint32_t opcode) {
    size_t left = 0;
    size_t right = len;
    while (left < right) {
        size_t middle = left + (right - left) / 2;
        if (keys[middle] < opcode) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__device__ size_t opcode_upper_bound(uint32_t const *keys, size_t len, uint32_t opcode) {
    size_t left = 0;
    size_t right = len;
    while (left < right) {
        size_t middle = left + (right - left) / 2;
        if (keys[middle] <= opcode) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__global__ void build_opcode_ranges(
    uint32_t const *sorted_keys,
    size_t num_steps,
    DeviceBufferConstView<uint32_t> active_opcodes,
    RvrOpcodeRange *ranges
) {
    size_t opcode_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (opcode_index >= active_opcodes.len()) return;
    uint32_t opcode = active_opcodes[opcode_index];
    ranges[opcode_index] = RvrOpcodeRange{
        .start = static_cast<uint32_t>(opcode_lower_bound(sorted_keys, num_steps, opcode)),
        .end = static_cast<uint32_t>(opcode_upper_bound(sorted_keys, num_steps, opcode)),
    };
}

} // namespace

extern "C" int _rvr_program_index_get_temp_bytes(
    size_t num_steps,
    size_t *h_temp_bytes_out,
    cudaStream_t stream
) {
    size_t temp_bytes = 0;
    if (num_steps != 0) {
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_bytes,
            static_cast<uint32_t *>(nullptr),
            static_cast<uint32_t *>(nullptr),
            static_cast<RvrReplayStep *>(nullptr),
            static_cast<RvrReplayStep *>(nullptr),
            num_steps,
            0,
            32,
            stream
        );
    }
    *h_temp_bytes_out = temp_bytes;
    return CHECK_KERNEL();
}

extern "C" int _rvr_program_index(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    DeviceBufferConstView<uint32_t> dense_program_rows,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint32_t> active_opcodes,
    uint32_t timestamp_max_bits,
    uint32_t endpoint_kind,
    uint32_t resume_pc,
    uint32_t final_timestamp,
    uint32_t terminate_opcode,
    uint32_t *opcode_keys_in,
    uint32_t *opcode_keys_out,
    RvrReplayStep *steps_in,
    RvrReplayStep *steps_out,
    RvrOpcodeRange *ranges,
    uint32_t *program_frequencies,
    void *temp_storage,
    size_t temp_storage_bytes,
    uint32_t *error,
    cudaStream_t stream
) {
    size_t num_steps = program.len() - 1;
    if (num_steps == 0) {
        validate_empty_program<<<1, 1, 0, stream>>>(
            instructions,
            pc_base,
            program,
            memory,
            timestamp_max_bits,
            endpoint_kind,
            resume_pc,
            final_timestamp,
            error
        );
        if (int err = CHECK_KERNEL(); err) return err;
        if (active_opcodes.len() != 0) {
            if (cudaError_t err = cudaMemsetAsync(
                    ranges, 0, 2 * active_opcodes.len() * sizeof(uint32_t), stream
                );
                err != cudaSuccess) {
                return err;
            }
        }
        return 0;
    }
    auto [step_grid, step_block] = kernel_launch_params(num_steps);
    prepare_program_steps<<<step_grid, step_block, 0, stream>>>(
        instructions,
        dense_program_rows,
        pc_base,
        program,
        memory,
        timestamp_max_bits,
        endpoint_kind,
        resume_pc,
        final_timestamp,
        terminate_opcode,
        opcode_keys_in,
        steps_in,
        program_frequencies,
        error
    );
    if (int err = CHECK_KERNEL(); err) return err;
    if (cudaError_t err = cub::DeviceRadixSort::SortPairs(
            temp_storage,
            temp_storage_bytes,
            opcode_keys_in,
            opcode_keys_out,
            steps_in,
            steps_out,
            num_steps,
            0,
            32,
            stream
        );
        err != cudaSuccess) {
        return err;
    }
    if (active_opcodes.len() != 0) {
        auto [range_grid, range_block] = kernel_launch_params(active_opcodes.len());
        build_opcode_ranges<<<range_grid, range_block, 0, stream>>>(
            opcode_keys_out, num_steps, active_opcodes, ranges
        );
        if (int err = CHECK_KERNEL(); err) return err;
    }
    return 0;
}
