#include "openvm.h"
#include "rvr_ext_sha2.h"

uint8_t rvr_ext_sha2_is_preflight(void) {
#ifdef OPENVM_TRACER_PREFLIGHT_H
  return 1;
#else
  return 0;
#endif
}
