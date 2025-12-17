#pragma once

#include "fp.h"

#include <cstdint>
#include <stddef.h>

constexpr size_t D_EF = 4;
constexpr size_t CHUNK = 8;
constexpr size_t WIDTH = 16;
constexpr size_t DIGEST_SIZE = 8;
typedef Fp Digest[DIGEST_SIZE];

typedef struct {
    size_t air_idx;
    size_t cached_idx;
    size_t starting_cidx;
    size_t total_interactions;
    size_t num_air_id_lookups;
    uint8_t log_height;
} TraceMetadata;

typedef struct {
    size_t air_idx;
    size_t air_num_pvs;
    size_t num_airs;
    size_t pv_idx;
    Fp value;
} PublicValueData;

typedef struct {
    size_t num_cached;
    size_t num_interactions_per_row;
    bool has_preprocessed;
} AirData;

typedef struct {
    uint16_t proof_idx;
    uint16_t merkle_proof_idx;
    uint32_t start_row;
    uint32_t num_rows;
    uint16_t depth;
    uint32_t merkle_idx;
    uint16_t commit_major;
    uint16_t commit_minor;
    uint32_t leaf_hash_offset;
    uint32_t siblings_offset;
} MerkleVerifyRecord;
