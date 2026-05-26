#ifndef OPENVM_UTIL_H
#define OPENVM_UTIL_H

static __attribute__((always_inline)) inline int likely(int x) {
  return __builtin_expect(!!(x), 1);
}

static __attribute__((always_inline)) inline int unlikely(int x) {
  return __builtin_expect(!!(x), 0);
}

#endif
