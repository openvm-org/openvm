#ifndef OPENVM_IO_H
#define OPENVM_IO_H

#include <stdint.h>

typedef void (*HintStreamSetFn)(void* ctx, const uint8_t* data, uint64_t len);

void register_openvm_io_ctx(void* ctx);
void register_hint_stream_set_fn(HintStreamSetFn fn);
void* openvm_get_io_ctx(void);

/* Replace the hint stream contents. Called by extension FFI staticlibs. */
void ext_hint_stream_set(const uint8_t* data, uint64_t len);

#endif /* OPENVM_IO_H */
