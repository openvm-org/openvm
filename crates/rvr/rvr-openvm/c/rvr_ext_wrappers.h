#ifndef RVR_EXT_WRAPPERS_H
#define RVR_EXT_WRAPPERS_H

#include <stdint.h>

#include "openvm_state.h"

uint64_t rd_mem_u64_wrapper(RvState* state, uint64_t addr);
void rd_mem_u64_range_wrapper(RvState* state, uint64_t base_addr, uint64_t* out,
                              uint32_t num_words);
void wr_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                              const uint64_t* vals, uint32_t num_words);
void trace_rd_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                    const uint64_t* vals, uint32_t num_words);
void trace_wr_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                    const uint64_t* vals, uint32_t num_words);
void trace_mem_access_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                        uint32_t num_words,
                                        uint32_t addr_space);
#endif /* RVR_EXT_WRAPPERS_H */
