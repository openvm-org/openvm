#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "sha256-air/columns.cuh"
#include "sha256-air/tracegen.cuh"
#include "sha256-air/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include <cassert>

using namespace riscv;
using namespace sha256;

__device__ inline void write_round_padding_flags_encoder(
    RowSlice row,
    const Encoder &padding_encoder,
    uint32_t flag_idx
) {
    RowSlice pad_flags = row.slice_from(COL_INDEX(Sha256VmRoundCols, control.pad_flags));
    padding_encoder.write_flag_pt(pad_flags, flag_idx);
}

__device__ inline void write_digest_padding_flags_encoder(
    RowSlice row,
    const Encoder &padding_encoder,
    uint32_t flag_idx
) {
    RowSlice pad_flags = row.slice_from(COL_INDEX(Sha256VmDigestCols, control.pad_flags));
    padding_encoder.write_flag_pt(pad_flags, flag_idx);
}

// ===== MAIN KERNEL FUNCTIONS =====
__global__ void sha256_hash_computation(
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    uint32_t *block_offsets,
    uint32_t *prev_hashes,
    uint32_t total_num_blocks
) {
    uint32_t record_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (record_idx >= num_records) {
        return;
    }

    uint32_t offset = record_offsets[record_idx];
    Sha256VmRecordMut record(records + offset);

    uint32_t len = record.header->len;
    uint8_t *input = record.input;

    uint32_t start_block = block_offsets[record_idx];
    uint32_t num_blocks = get_sha256_num_blocks(len);

    uint32_t current_hash[SHA256_HASH_WORDS] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };

    uint8_t block_input[SHA256_BLOCK_U8S];
    for (uint32_t local_block = 0; local_block < num_blocks; local_block++) {
        {
            uint32_t offset = SHA256_HASH_WORDS * (start_block + local_block);
            memcpy(prev_hashes + offset, current_hash, SHA256_HASH_WORDS * sizeof(uint32_t));
        }
        if (local_block < num_blocks - 1) {
            // Since local_block < num_blocks - 1, we know that block_offset < len by the definition of num_blocks
            uint32_t block_offset = local_block * SHA256_BLOCK_U8S;
            if (block_offset + SHA256_BLOCK_U8S > len) {
                uint32_t remaining_bytes = len - block_offset;
                memcpy(block_input, input + block_offset, remaining_bytes);
                block_input[remaining_bytes] = 0x80;
                memset(
                    block_input + remaining_bytes + 1, 0, SHA256_BLOCK_U8S - remaining_bytes - 1
                );
            } else {
                memcpy(block_input, input + block_offset, SHA256_BLOCK_U8S);
            }

            get_block_hash(current_hash, block_input);
        }
    }
}

__global__ __noinline__ void sha256_first_pass_tracegen(
    Fp *trace,
    size_t trace_height,
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    uint32_t *block_offsets,
    uint32_t *block_to_record_idx,
    uint32_t total_num_blocks,
    uint32_t *prev_hashes,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t global_block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_block_idx >= total_num_blocks) {
        return;
    }

    uint32_t record_idx = block_to_record_idx[global_block_idx];
    if (record_idx >= num_records) {
        return;
    }

    uint32_t offset = record_offsets[record_idx];
    Sha256VmRecordMut record(records + offset);
    Sha256VmRecordHeader &vm_record = *record.header;

    auto len = vm_record.len;
    auto input = record.input;

    auto local_block_idx = global_block_idx - block_offsets[record_idx];
    auto prev_hash = &prev_hashes[global_block_idx * SHA256_HASH_WORDS];
    auto block_offset = local_block_idx * SHA256_BLOCK_U8S;

    uint32_t num_blocks_for_record = get_sha256_num_blocks(len);
    bool is_last_block = (local_block_idx == num_blocks_for_record - 1);

    uint32_t input_words[SHA256_BLOCK_WORDS];
    {
        uint8_t padded_input[SHA256_BLOCK_U8S];
        if (block_offset <= len) {
            if (block_offset + SHA256_BLOCK_U8S > len) {
                uint32_t remaining_bytes = len - block_offset;
                memcpy(padded_input, input + block_offset, remaining_bytes);
                padded_input[remaining_bytes] = 0x80;
                memset(
                    padded_input + remaining_bytes + 1, 0, SHA256_BLOCK_U8S - remaining_bytes - 1
                );
            } else {
                memcpy(padded_input, input + block_offset, SHA256_BLOCK_U8S);
            }
        } else {
            memset(padded_input, 0, SHA256_BLOCK_U8S);
        }

        for (uint32_t i = 0; i < SHA256_BLOCK_WORDS; i++) {
            input_words[i] = u32_from_bytes_be(padded_input + i * 4);
        }

        if (is_last_block) {
            input_words[SHA256_BLOCK_WORDS - 1] = len << 3;
        }
    }

    uint32_t trace_start_row = global_block_idx * SHA256_ROWS_PER_BLOCK;

    uint32_t read_cells = (SHA256_BLOCK_U8S * local_block_idx);
    uint32_t block_start_read_ptr = vm_record.src_ptr + read_cells;
    uint32_t message_left = len - read_cells;

    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
    );

    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);

    Encoder padding_encoder(static_cast<uint32_t>(PaddingFlags::COUNT), 2, false);

    int32_t first_padding_row;
    if (len < read_cells) {
        first_padding_row = -1;
    } else if (message_left < SHA256_BLOCK_U8S) {
        first_padding_row = message_left / SHA256_READ_SIZE;
    } else {
        first_padding_row = 18;
    }

    auto start_timestamp =
        vm_record.timestamp + (SHA256_REGISTER_READS + SHA256_NUM_READ_ROWS * local_block_idx);

    for (uint32_t row_in_block = 0; row_in_block < SHA256_ROWS_PER_BLOCK; row_in_block++) {
        uint32_t absolute_row = trace_start_row + row_in_block;

        if (absolute_row >= trace_height) {
            return;
        }

        RowSlice row(trace + absolute_row, trace_height);

        if (row_in_block == SHA256_ROWS_PER_BLOCK - 1) {
            SHA256_WRITE_DIGEST(row, from_state.timestamp, vm_record.timestamp);
            SHA256_WRITE_DIGEST(row, from_state.pc, vm_record.from_pc);
            SHA256_WRITE_DIGEST(row, rd_ptr, vm_record.rd_ptr);
            SHA256_WRITE_DIGEST(row, rs1_ptr, vm_record.rs1_ptr);
            SHA256_WRITE_DIGEST(row, rs2_ptr, vm_record.rs2_ptr);

            {
                uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(&vm_record.dst_ptr);
                uint8_t *src_bytes = reinterpret_cast<uint8_t *>(&vm_record.src_ptr);
                uint8_t *len_bytes = reinterpret_cast<uint8_t *>(&len);

                SHA256_WRITE_ARRAY_DIGEST(row, dst_ptr, dst_bytes);
                SHA256_WRITE_ARRAY_DIGEST(row, src_ptr, src_bytes);
                SHA256_WRITE_ARRAY_DIGEST(row, len_data, len_bytes);
            }

            if (is_last_block) {
                for (int i = 0; i < SHA256_REGISTER_READS; i++) {
                    mem_helper.fill(
                        SHA256_SLICE_DIGEST(row, register_reads_aux[i]),
                        vm_record.register_reads_aux[i].prev_timestamp,
                        vm_record.timestamp + i
                    );
                }

                SHA256_WRITE_ARRAY_DIGEST(row, writes_aux.prev_data, vm_record.write_aux.prev_data);

                mem_helper.fill(
                    SHA256_SLICE_DIGEST(row, writes_aux),
                    vm_record.write_aux.prev_timestamp,
                    start_timestamp + SHA256_NUM_READ_ROWS
                );

                uint32_t msl_rshift = ((RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS);
                uint32_t msl_lshift = (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - ptr_max_bits);
                bitwise_lookup.add_range(
                    (vm_record.dst_ptr >> msl_rshift) << msl_lshift,
                    (vm_record.src_ptr >> msl_rshift) << msl_lshift
                );
            } else {
                for (int i = 0; i < SHA256_REGISTER_READS; i++) {
                    mem_helper.fill_zero(SHA256_SLICE_DIGEST(row, register_reads_aux[i]));
                }

                row.fill_zero(
                    COL_INDEX(Sha256VmDigestCols, writes_aux.prev_data), SHA256_WRITE_SIZE
                );
                mem_helper.fill_zero(SHA256_SLICE_DIGEST(row, writes_aux));
            }
            SHA256_WRITE_DIGEST(row, inner.flags.is_last_block, is_last_block);
            SHA256_WRITE_DIGEST(row, inner.flags.is_digest_row, Fp::one());
            row.fill_zero(SHA256VM_DIGEST_WIDTH, SHA256VM_WIDTH - SHA256VM_DIGEST_WIDTH);
        } else {
            if (row_in_block < SHA256_NUM_READ_ROWS) {

                uint32_t data_offset = block_offset + row_in_block * SHA256_READ_SIZE;
                SHA256_WRITE_ARRAY_ROUND(
                    row, inner.message_schedule.carry_or_buffer, input + data_offset
                );

                MemoryReadAuxRecord *read_aux_record =
                    get_read_aux_record(&record, local_block_idx, row_in_block);
                mem_helper.fill(
                    SHA256_SLICE_ROUND(row, read_aux),
                    read_aux_record->prev_timestamp,
                    start_timestamp + row_in_block
                );
            } else {
                mem_helper.fill_zero(SHA256_SLICE_ROUND(row, read_aux));
            }
        }

        SHA256_WRITE_ROUND(row, control.len, len);

        {
            uint32_t control_timestamp =
                start_timestamp + min(row_in_block, (uint32_t)SHA256_NUM_READ_ROWS);
            uint32_t control_read_ptr =
                block_start_read_ptr +
                (SHA256_READ_SIZE * min(row_in_block, (uint32_t)SHA256_NUM_READ_ROWS));

            SHA256_WRITE_ROUND(row, control.cur_timestamp, control_timestamp);
            SHA256_WRITE_ROUND(row, control.read_ptr, control_read_ptr);
        }

        if (row_in_block < SHA256_NUM_READ_ROWS) {
            if ((int32_t)row_in_block < first_padding_row) {
                write_round_padding_flags_encoder(
                    row, padding_encoder, static_cast<uint32_t>(PaddingFlags::NotPadding)
                );
            } else if ((int32_t)row_in_block == first_padding_row) {
                {
                    uint32_t len = message_left - row_in_block * SHA256_READ_SIZE;
                    uint32_t flag_idx;
                    if (row_in_block == 3 && is_last_block) {
                        flag_idx = static_cast<uint32_t>(PaddingFlags::FirstPadding0_LastRow) + len;
                        if (flag_idx >= static_cast<uint32_t>(PaddingFlags::COUNT)) {
                            flag_idx = static_cast<uint32_t>(PaddingFlags::EntirePaddingLastRow);
                        }
                    } else {
                        flag_idx = static_cast<uint32_t>(PaddingFlags::FirstPadding0) + len;
                        if (flag_idx >= static_cast<uint32_t>(PaddingFlags::COUNT)) {
                            flag_idx = static_cast<uint32_t>(PaddingFlags::EntirePadding);
                        }
                    }
                    write_round_padding_flags_encoder(row, padding_encoder, flag_idx);
                }
            } else {
                {
                    uint32_t flag_idx;
                    if (row_in_block == 3 && is_last_block) {
                        flag_idx = static_cast<uint32_t>(PaddingFlags::EntirePaddingLastRow);
                    } else {
                        flag_idx = static_cast<uint32_t>(PaddingFlags::EntirePadding);
                    }
                    write_round_padding_flags_encoder(row, padding_encoder, flag_idx);
                }
            }
        } else {
            write_round_padding_flags_encoder(
                row, padding_encoder, static_cast<uint32_t>(PaddingFlags::NotConsidered)
            );
        }

        if (is_last_block && row_in_block == SHA256_ROWS_PER_BLOCK - 1) {
            SHA256_WRITE_ROUND(row, control.padding_occurred, Fp::zero());
        } else {
            SHA256_WRITE_ROUND(
                row, control.padding_occurred, (int32_t)row_in_block >= first_padding_row
            );
        }
    }

    Fp *inner_trace_start = trace + (SHA256_INNER_COLUMN_OFFSET * trace_height) + trace_start_row;
    generate_block_trace(
        inner_trace_start,
        trace_height,
        input_words,
        bitwise_lookup_ptr,
        bitwise_num_bits,
        prev_hash,
        is_last_block,
        global_block_idx + 1,
        local_block_idx
    );
}

__global__ void sha256_fill_invalid_rows(Fp *d_trace, size_t trace_height, size_t rows_used) {
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row_idx = rows_used + thread_idx;
    if (row_idx >= trace_height) {
        return;
    }

    RowSlice row(d_trace + row_idx, trace_height);
    row.fill_zero(0, SHA256VM_WIDTH);

    RowSlice inner_row = row.slice_from(SHA256_INNER_COLUMN_OFFSET);
    generate_default_row(inner_row);
}

// ===== HOST LAUNCHER FUNCTIONS =====
extern "C" int launch_sha256_hash_computation(
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t *d_block_offsets,
    uint32_t *d_prev_hashes,
    uint32_t total_num_blocks
) {
    auto [grid_size, block_size] = kernel_launch_params(num_records, 256);

    sha256_hash_computation<<<grid_size, block_size>>>(
        d_records, num_records, d_record_offsets, d_block_offsets, d_prev_hashes, total_num_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int launch_sha256_first_pass_tracegen(
    Fp *d_trace,
    size_t trace_height,
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t *d_block_offsets,
    uint32_t *d_block_to_record_idx,
    uint32_t total_num_blocks,
    uint32_t *d_prev_hashes,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    // Validate that trace_height is a power of two
    assert((trace_height & (trace_height - 1)) == 0);
    assert(trace_height >= total_num_blocks * SHA256_ROWS_PER_BLOCK);

    auto [grid_size, block_size] = kernel_launch_params(total_num_blocks, 256);

    sha256_first_pass_tracegen<<<grid_size, block_size>>>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        d_block_offsets,
        d_block_to_record_idx,
        total_num_blocks,
        d_prev_hashes,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );

    return CHECK_KERNEL();
}

extern "C" int launch_sha256_second_pass_dependencies(
    Fp *d_trace,
    size_t trace_height,
    size_t rows_used
) {
    Fp *inner_trace_start = d_trace + (SHA256_INNER_COLUMN_OFFSET * trace_height);
    uint32_t total_sha256_blocks = rows_used / SHA256_ROWS_PER_BLOCK;

    auto [grid_size, block_size] = kernel_launch_params(total_sha256_blocks, 256);

    sha256_second_pass_dependencies<<<grid_size, block_size>>>(
        inner_trace_start, trace_height, total_sha256_blocks
    );

    return CHECK_KERNEL();
}

extern "C" int launch_sha256_fill_invalid_rows(
    Fp *d_trace,
    size_t trace_height,
    size_t rows_used
) {
    uint32_t invalid_rows = trace_height - rows_used;
    auto [grid_size, block_size] = kernel_launch_params(invalid_rows, 256);
    sha256_fill_invalid_rows<<<grid_size, block_size>>>(d_trace, trace_height, rows_used);

    return CHECK_KERNEL();
}