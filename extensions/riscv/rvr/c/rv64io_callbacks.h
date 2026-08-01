#ifndef RV64IO_CALLBACKS_H
#define RV64IO_CALLBACKS_H

#include <stdint.h>

typedef struct {
  uint64_t block_addr;
  uint64_t previous[2];
  uint64_t post[2];
  uint8_t crosses;
  uint8_t _padding[7];
} Rv64RevealPlan;

static_assert(sizeof(Rv64RevealPlan) == 48);
static_assert(alignof(Rv64RevealPlan) == 8);

typedef struct {
  bool (*hint_prepare)(void* ctx, uint64_t dest_addr, uint32_t num_words);
  void (*hint_read_words)(void* ctx, uint64_t* words, uint32_t num_words);
  bool (*hint_storew)(void* ctx, uint64_t dest_addr);
  bool (*hint_buffer)(void* ctx, uint64_t dest_addr, uint32_t num_words);
  bool (*reveal_prepare)(void* ctx, uint64_t src_val, uint64_t base_addr,
                         uint64_t effective_addr, uint8_t width,
                         Rv64RevealPlan* plan);
  void (*reveal_commit)(void* ctx, const Rv64RevealPlan* plan);
} Rv64IoHostCallbacks;

/* Callbacks and forwarding stubs return false for invalid guest operands. */

void register_rv64io_host_callbacks(const Rv64IoHostCallbacks* cb);

/* Forwarding stubs owned by the RISC-V IO (Rv64Io) extension: validation and
 * consumption for HINT_STOREW/HINT_BUFFER, plus public-values stores routed
 * through openvm_reveal. Backed by a thread-local dispatch table installed at
 * execution time by `Rv64IoRuntimeHooks`. */
bool openvm_hint_prepare(uint64_t dest_addr, uint32_t num_words);
void openvm_hint_read_words(uint64_t* words, uint32_t num_words);
bool openvm_hint_storew(uint64_t dest_addr);
bool openvm_hint_buffer(uint64_t dest_addr, uint32_t num_words);
bool openvm_reveal(RvState* state, uint64_t src_val, uint64_t base_addr,
                   uint64_t effective_addr, uint8_t width);

#endif /* RV64IO_CALLBACKS_H */
