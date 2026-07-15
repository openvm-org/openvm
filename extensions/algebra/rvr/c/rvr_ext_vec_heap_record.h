#ifndef RVR_EXT_VEC_HEAP_RECORD_H
#define RVR_EXT_VEC_HEAP_RECORD_H

#include <stdint.h>
#include <string.h>

struct RvState;

#ifdef OPENVM_TRACER_PREFLIGHT_H
typedef struct RvrVecHeapRecordDescriptor {
  uint32_t num_reads;
  uint32_t blocks;
  uint32_t adapter_size;
  uint32_t core_size;
  uint32_t record_size;
  uint32_t reads_aux;
  uint32_t writes_aux;
} RvrVecHeapRecordDescriptor;

static inline RvrVecHeapRecordDescriptor rvr_vec_heap_descriptor(
    uint32_t num_limbs, uint32_t num_reads) {
  static constexpr uint32_t word_size = 8;
  uint32_t blocks = num_limbs / word_size;
  uint32_t reads_aux = 20u + 12u * num_reads;
  uint32_t writes_aux = reads_aux + 4u * num_reads * blocks;
  uint32_t adapter_size = writes_aux + 12u * blocks;
  uint32_t core_size = 1u + 8u * num_reads * blocks;
  return (RvrVecHeapRecordDescriptor){
      .num_reads = num_reads,
      .blocks = blocks,
      .adapter_size = adapter_size,
      .core_size = core_size,
      .record_size = (adapter_size + core_size + 3u) & ~3u,
      .reads_aux = reads_aux,
      .writes_aux = writes_aux,
  };
}

static inline void rvr_store_u64_unaligned_le(uint8_t* dst, uint64_t value) {
  memcpy(dst, &value, sizeof(value));
}

static inline uint8_t* rvr_claim_vec_heap_record(
    RvState* state, uint32_t chip_idx, uint32_t compact_core_off,
    uint32_t record_size, uint8_t** core) {
  uint8_t* record = (uint8_t*)preflight_claim_record(state, chip_idx);
  if (unlikely(record == NULL)) {
    return NULL;
  }
  ChipRecordBuf* buf = &state->tracer->chip_records[chip_idx];
  if (unlikely(buf->stride < record_size)) {
    buf->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return NULL;
  }
  if ((buf->flags & PREFLIGHT_RECORD_DIRECT_FINAL) == 0u) {
    memset(record, 0, record_size);
  }
  uint32_t core_off = buf->core_off == 0u ? compact_core_off : buf->core_off;
  *core = record + core_off;
  return record;
}
#endif

static inline void rvr_ext_emit_vec_heap_record(
    RvState* state, uint32_t from_pc, uint32_t local_opcode,
    uint32_t num_limbs, uint32_t num_reads, uint32_t chip_idx) {
#ifdef OPENVM_TRACER_PREFLIGHT_H
  RvrVecHeapRecordDescriptor d =
      rvr_vec_heap_descriptor(num_limbs, num_reads);
  uint32_t register_events = d.num_reads + 1u;
  uint32_t event_count = register_events +
                         d.num_reads * d.blocks + d.blocks;
  Tracer* tracer = state->tracer;
  MemoryLogEntry* events =
      preflight_take_custom_memory_events(tracer, event_count);
  if (unlikely(events == NULL)) {
    return;
  }
  uint8_t* core;
  uint8_t* record = rvr_claim_vec_heap_record(
      state, chip_idx, d.adapter_size, d.record_size, &core);
  if (unlikely(record == NULL)) {
    return;
  }

  *(uint32_t*)(record + 0u) = from_pc;
  *(uint32_t*)(record + 4u) = events[0].timestamp;
  uint32_t rs_ptrs = 8u;
  uint32_t rd_ptr = rs_ptrs + 4u * d.num_reads;
  uint32_t rs_vals = rd_ptr + 4u;
  uint32_t rd_val = rs_vals + 4u * d.num_reads;
  uint32_t rs_read_aux = rd_val + 4u;
  uint32_t rd_read_aux = rs_read_aux + 4u * d.num_reads;
  for (uint32_t i = 0; i < d.num_reads; i++) {
    *(uint32_t*)(record + rs_ptrs + 4u * i) =
        (uint32_t)events[i].address;
    *(uint32_t*)(record + rs_vals + 4u * i) =
        (uint32_t)events[i].value;
    preflight_store_prev_timestamp(
        tracer, (uint32_t*)(record + rs_read_aux + 4u * i),
        events[i].prev_timestamp);
  }
  MemoryLogEntry* rd = &events[d.num_reads];
  *(uint32_t*)(record + rd_ptr) = (uint32_t)rd->address;
  *(uint32_t*)(record + rd_val) = (uint32_t)rd->value;
  preflight_store_prev_timestamp(
      tracer, (uint32_t*)(record + rd_read_aux), rd->prev_timestamp);

  uint32_t heap_start = register_events;
  for (uint32_t read = 0; read < d.num_reads; read++) {
    for (uint32_t block = 0; block < d.blocks; block++) {
      uint32_t idx = heap_start + read * d.blocks + block;
      uint32_t flat = read * d.blocks + block;
      preflight_store_prev_timestamp(
          tracer, (uint32_t*)(record + d.reads_aux + 4u * flat),
          events[idx].prev_timestamp);
      preflight_store_prev_value(tracer, core + 1u + 8u * flat,
                                 events[idx].prev_timestamp,
                                 events[idx].prev_value);
    }
  }
  uint32_t write_start = heap_start + d.num_reads * d.blocks;
  for (uint32_t block = 0; block < d.blocks; block++) {
    uint8_t* aux = record + d.writes_aux + 12u * block;
    MemoryLogEntry* write = &events[write_start + block];
    preflight_store_prev_timestamp(tracer, (uint32_t*)aux,
                                   write->prev_timestamp);
    preflight_store_prev_value(tracer, aux + 4u, write->prev_timestamp,
                               write->prev_value);
  }
  *core = (uint8_t)local_opcode;
#endif
}

#endif /* RVR_EXT_VEC_HEAP_RECORD_H */
