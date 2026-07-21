#ifndef OPENVM_UTIL_H
#define OPENVM_UTIL_H

#include <assert.h>

static __attribute__((always_inline)) inline int likely(int x) {
  return __builtin_expect(!!(x), 1) != 0;
}

static __attribute__((always_inline)) inline int unlikely(int x) {
  return __builtin_expect(!!(x), 0) != 0;
}

static __attribute__((always_inline)) inline void assume(int x) {
  __builtin_assume(x);
}

static __attribute__((always_inline)) inline void debug_assume(int x) {
  assert(x);
  assume(x);
}

#endif
