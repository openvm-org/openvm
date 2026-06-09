/*
 * IO ctx slot. The host installs a pointer via `register_openvm_io_ctx`;
 * extensions read it back via `openvm_get_io_ctx`.
 */

#include "openvm_io.h"

/* NOT thread-safe: installs one process-global IO ctx pointer. */
static void* g_io_ctx;

void register_openvm_io_ctx(void* ctx) { g_io_ctx = ctx; }

void* openvm_get_io_ctx(void) { return g_io_ctx; }
