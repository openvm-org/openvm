#pragma once

#include "fp.h"

#include <stddef.h>

const size_t DIGEST_SIZE = 8;
typedef Fp Digest[DIGEST_SIZE];

typedef struct {
  size_t hypercube_dim;
  bool is_present;
  size_t num_cached;
  size_t cached_idx;
} TraceMetadata;

typedef struct {
  size_t air_idx;
  size_t pv_idx;
  Fp value;
} PublicValueData;

typedef struct {
  size_t num_interactions;
  size_t has_preprocessed;
} AirData;
