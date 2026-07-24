/* Native-detail profiling helpers, included only by instrumented preflight
 * projects. ABI types and phase constants are defined by the common header. */

#ifndef OPENVM_TRACER_PREFLIGHT_NATIVE_DETAIL_H
#define OPENVM_TRACER_PREFLIGHT_NATIVE_DETAIL_H

static __attribute__((always_inline)) inline uint64_t
preflight_detail_clock(void) {
#if defined(__x86_64__)
  _mm_lfence();
  return __rdtsc();
#elif defined(__aarch64__)
  uint64_t value;
  __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(value));
  return value;
#else
  return 0u;
#endif
}

/* Sample one out of roughly 512-1535 phase events. The randomized countdown
 * avoids locking onto a guest loop period while leaving unsampled operations
 * with only a decrement and predictable branch in profiling builds. */
static __attribute__((noinline)) bool preflight_detail_phase_select(
    Tracer* restrict t, uint32_t phase, uint64_t bytes) {
  RvrNativeDetail* restrict detail = t->native_detail;
  if (unlikely(detail == NULL || phase >= PREFLIGHT_DETAIL_PHASE_COUNT)) {
    return false;
  }
  detail->phase_events[phase]++;
  detail->phase_bytes[phase] += bytes;
  uint32_t countdown = detail->sample_countdown;
  if (likely(countdown > 1u)) {
    detail->sample_countdown = countdown - 1u;
    return false;
  }
  uint32_t state = detail->sample_state * 1664525u + 1013904223u;
  detail->sample_state = state;
  detail->sample_countdown = 512u + (state & 1023u);
  detail->phase_samples[phase]++;
  return true;
}

static __attribute__((always_inline)) inline uint64_t
preflight_detail_phase_begin(Tracer* restrict t, uint32_t phase,
                             uint64_t bytes) {
  if (!OPENVM_RVR_NATIVE_DETAIL_ENABLED) {
    return 0u;
  }
  if (likely(!preflight_detail_phase_select(t, phase, bytes))) {
    return 0u;
  }
  return preflight_detail_clock();
}

static __attribute__((always_inline)) inline void preflight_detail_phase_end(
    Tracer* restrict t, uint32_t phase, uint64_t started) {
  if (!OPENVM_RVR_NATIVE_DETAIL_ENABLED) {
    return;
  }
  if (likely(started == 0u)) {
    return;
  }
  uint64_t finished = preflight_detail_clock();
  t->native_detail->phase_cycles[phase] += finished - started;
}

/* Inclusive opcode-family timing uses run boundaries instead of two clocks per
 * instruction. Long RV64IM runs therefore pay no cycle-clock overhead; custom
 * callbacks naturally include time spent across the host FFI boundary. */
static __attribute__((always_inline)) inline void preflight_detail_family(
    Tracer* restrict t, uint32_t family) {
  if (!OPENVM_RVR_NATIVE_DETAIL_ENABLED) {
    return;
  }
  RvrNativeDetail* restrict detail = t->native_detail;
  if (unlikely(detail == NULL || family >= PREFLIGHT_DETAIL_FAMILY_COUNT)) {
    return;
  }
  detail->family_instructions[family]++;
  if (likely(detail->family_active != 0u &&
             detail->current_family == family)) {
    return;
  }
  uint64_t now = preflight_detail_clock();
  if (detail->family_active != 0u) {
    detail->family_cycles[detail->current_family] +=
        now - detail->family_started;
  }
  detail->current_family = family;
  detail->family_started = now;
  detail->family_active = 1u;
}

#endif /* OPENVM_TRACER_PREFLIGHT_NATIVE_DETAIL_H */
