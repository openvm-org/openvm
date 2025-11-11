#pragma once

#include "fp.h"

#include <cstdint>
#include <stddef.h>

const size_t DIGEST_SIZE = 8;
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
