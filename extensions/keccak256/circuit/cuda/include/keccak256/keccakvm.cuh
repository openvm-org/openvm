#pragma once

#include "primitives/constants.h"
#include "primitives/utils.cuh"
#include "system/memory/offline_checker.cuh"

using namespace keccak256;

__device__ __forceinline__ size_t num_keccak_f(size_t byte_len) {
    return (byte_len / KECCAK_RATE_BYTES) + 1;
}

struct KeccakVmRecordHeader {
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint32_t dst;
    uint32_t src;
    uint32_t len;

    MemoryReadAuxRecord register_reads_aux[KECCAK_REGISTER_READS];
    MemoryWriteBytesAuxRecord<KECCAK_WORD_SIZE> write_aux[KECCAK_DIGEST_WRITES];
};

struct KeccakVmRecordMut {
    KeccakVmRecordHeader header;
    uint8_t *input_ptr;
    MemoryReadAuxRecord *read_aux_ptr;

    __device__ __host__ __forceinline__ KeccakVmRecordMut(uint8_t *record_buf) {
        header = *reinterpret_cast<KeccakVmRecordHeader *>(record_buf);
        input_ptr = record_buf + sizeof(KeccakVmRecordHeader);
        auto read_aux_offset = next_multiple_of(read_len(), alignof(MemoryReadAuxRecord));
        read_aux_ptr = reinterpret_cast<MemoryReadAuxRecord *>(input_ptr + read_aux_offset);
    }

    __device__ __host__ __forceinline__ size_t num_reads() const {
        return d_div_ceil(size_t(header.len), KECCAK_WORD_SIZE);
    }

    __device__ __host__ __forceinline__ size_t read_len() const {
        return num_reads() * KECCAK_WORD_SIZE;
    }
};
