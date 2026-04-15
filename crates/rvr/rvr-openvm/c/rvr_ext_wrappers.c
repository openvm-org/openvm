/*
 * rvr_ext_wrappers.c — Non-inline wrappers around static inline tracing
 * functions.
 *
 * Extension FFI code (implemented in Rust as staticlibs) calls these wrappers
 * for traced memory access, chip cost, and block metering. The wrappers
 * delegate to the static inline functions defined in the tracer headers.
 * Register access stays in generated C; extensions receive resolved register
 * values as function parameters.
 *
 * This file is only compiled when extensions are present.
 */

#include "openvm.h"

/* ── Memory access (single word) ───────────────────────────────────── */

void trace_mem_access_u32_wrapper(RvState* s, uint32_t addr, uint32_t addr_space) {
  trace_mem_access(s, addr, addr_space);
}

uint32_t rd_mem_u32_wrapper(RvState* s, uint32_t addr) { return rd_mem_u32(s, addr); }
void wr_mem_u32_wrapper(RvState* s, uint32_t addr, uint32_t val) { wr_mem_u32(s, addr, val); }
void trace_rd_mem_u32_wrapper(RvState* s, uint32_t addr, uint32_t val) {
  trace_rd_mem_u32(s, addr, val);
}
void trace_wr_mem_u32_wrapper(RvState* s, uint32_t addr, uint32_t val) {
  trace_wr_mem_u32(s, addr, val);
}

/* ── Memory access (word-aligned ranges, inline-backed) ────────────── */

void rd_mem_u32_range_wrapper(RvState* s, uint32_t base_addr, uint32_t* out, uint32_t num_words) {
  rd_mem_u32_range(s, base_addr, out, num_words);
}

void wr_mem_u32_range_wrapper(RvState* s, uint32_t base_addr, const uint32_t* vals,
                              uint32_t num_words) {
  wr_mem_u32_range(s, base_addr, vals, num_words);
}

void trace_rd_mem_u32_range_wrapper(RvState* s, uint32_t base_addr, const uint32_t* vals,
                                    uint32_t num_words) {
  trace_rd_mem_u32_range(s, base_addr, vals, num_words);
}

void trace_wr_mem_u32_range_wrapper(RvState* s, uint32_t base_addr, const uint32_t* vals,
                                    uint32_t num_words) {
  trace_wr_mem_u32_range(s, base_addr, vals, num_words);
}

void trace_mem_access_u32_range_wrapper(RvState* s, uint32_t base_addr, uint32_t num_words,
                                        uint32_t addr_space) {
  trace_mem_access_u32_range(s, base_addr, num_words, addr_space);
}

/* ── Instruction dispatch / chip cost ──────────────────────────────── */

void trace_pc_wrapper(RvState* s, uint32_t pc) { trace_pc(s, pc); }

/* Extension FFI staticlibs use this to add chip rows on top of what the
 * per-block chip accounting in the generated `block_*` functions has already
 * counted.
 *
 * The block-entry update assigns +1 per instruction to that instruction's PC
 * chip (from `pc_to_chip`). So when an extension call needs to add MORE rows
 * to *its own* PC chip, it should pass `count - 1` (e.g. deferral OUTPUT).
 * When it needs to add rows to a DIFFERENT chip — which is the common case
 * for inner sub-chips like Sha2BlockHasher or the Poseidon2 chip used by
 * deferral CALL/OUTPUT — it should pass the full count for that other chip,
 * since the block-entry update never touched it.
 *
 * Extensions whose entry points are compiled directly into the generated
 * native project (e.g. `crates/extensions/keccak/c/rvr_ext_keccak.c`) call
 * the inline `trace_chip` directly; only out-of-line staticlib FFIs need
 * this wrapper. See callers in
 * `crates/extensions/{sha2,deferral}/ffi/src/lib.rs`. */
void trace_chip_wrapper(RvState* s, uint32_t chip_idx, uint32_t count) {
  trace_chip(s, chip_idx, count);
}

/* ── Block metering ────────────────────────────────────────────────── */

void trace_block_wrapper(RvState* s, uint32_t pc, uint32_t block_insn_count) {
  trace_block(s, pc, block_insn_count);
}
