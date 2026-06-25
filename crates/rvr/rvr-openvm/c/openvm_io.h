#ifndef OPENVM_IO_H
#define OPENVM_IO_H

#include <stdint.h>

void* openvm_get_io_ctx(void);

/* Called once at execute-start to install the field-typed hint-stream writer. */
void register_hint_stream_set_fn(void (*fn)(void*, const uint8_t*, uint32_t));

/* Replace the hint stream contents. Called by extension FFI staticlibs. */
void ext_hint_stream_set(const uint8_t* data, uint32_t len);

#endif /* OPENVM_IO_H */
