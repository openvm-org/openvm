#ifndef OPENVM_IO_H
#define OPENVM_IO_H

#include <stdint.h>

/* OpenVM IO runtime declarations */
void openvm_hint_input(void);
void openvm_print_str(uint32_t ptr, uint32_t len);
void openvm_hint_random(uint32_t num_words);
void openvm_hint_storew(uint32_t dest_addr);
void openvm_hint_buffer(uint32_t dest_addr, uint32_t num_words);
void openvm_reveal(uint32_t src_val, uint32_t ptr, uint32_t offset);

/* Extension hint stream access (called by extension FFI staticlibs). */
void ext_hint_stream_set(const uint8_t* data, uint32_t len);

/* Deferral extension callbacks (called by deferral FFI staticlib). */
int ext_deferral_call_lookup(uint32_t def_idx, const uint8_t* input_commit,
                             uint8_t* output_key_out);
int ext_deferral_output_lookup(const uint8_t* output_commit, uint8_t* output_raw_out,
                               uint32_t expected_len);

#endif /* OPENVM_IO_H */
