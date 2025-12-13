#pragma once

#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"
#include "variant.cuh"

namespace sha2 {

// GPU view of the per-block record produced by the executor (matches Sha2RecordMut in Rust).
template <typename V> struct Sha2BlockRecordHeader {
    uint32_t variant;
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t dst_reg_ptr;
    uint32_t state_reg_ptr;
    uint32_t input_reg_ptr;
    uint32_t dst_ptr;
    uint32_t state_ptr;
    uint32_t input_ptr;
    MemoryReadAuxRecord register_reads_aux[sha2::SHA2_REGISTER_READS];
};

template <typename V> struct Sha2BlockRecordMut {
    Sha2BlockRecordHeader<V> *header;
    uint8_t *message_bytes;
    uint8_t *prev_state;
    uint8_t *new_state;
    MemoryReadAuxRecord *input_reads_aux;
    MemoryReadAuxRecord *state_reads_aux;
    MemoryWriteBytesAuxRecord<sha2::SHA2_WRITE_SIZE> *write_aux;

    __device__ __host__ __forceinline__ static uint32_t next_multiple_of(
        uint32_t value,
        uint32_t alignment
    ) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    __device__ __host__ __forceinline__ Sha2BlockRecordMut(uint8_t *record_buf) {
        header = reinterpret_cast<Sha2BlockRecordHeader<V> *>(record_buf);
        uint32_t offset = sizeof(Sha2BlockRecordHeader<V>);

        message_bytes = record_buf + offset;
        offset += V::BLOCK_U8S;

        prev_state = record_buf + offset;
        offset += V::STATE_BYTES;

        new_state = record_buf + offset;
        offset += V::STATE_BYTES;

        offset = next_multiple_of(offset, alignof(MemoryReadAuxRecord));
        input_reads_aux = reinterpret_cast<MemoryReadAuxRecord *>(record_buf + offset);
        offset += V::BLOCK_READS * sizeof(MemoryReadAuxRecord);

        offset = next_multiple_of(offset, alignof(MemoryReadAuxRecord));
        state_reads_aux = reinterpret_cast<MemoryReadAuxRecord *>(record_buf + offset);
        offset += V::STATE_READS * sizeof(MemoryReadAuxRecord);

        offset = next_multiple_of(
            offset, alignof(MemoryWriteBytesAuxRecord<sha2::SHA2_WRITE_SIZE>)
        );
        write_aux = reinterpret_cast<MemoryWriteBytesAuxRecord<sha2::SHA2_WRITE_SIZE> *>(
            record_buf + offset
        );
    }
};

} // namespace sha2
