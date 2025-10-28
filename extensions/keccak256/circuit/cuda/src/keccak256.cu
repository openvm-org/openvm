#include "keccak256/columns.cuh"
#include "keccak256/keccakvm.cuh"
#include "keccak256/p3_generation.cuh"
#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"

using namespace keccak256;

__global__ void p3_inner_tracegen(
    Fp *trace,
    size_t height,
    uint32_t total_num_blocks, // actual number of blocks (rows_used / ROUNDS)
    uint32_t blocks_to_fill,   // includes dummy rows (1 + (height / ROUNDS))
    uint64_t *states
) {
    auto block_idx = blockIdx.x * blockDim.x + threadIdx.x; // [0 .. blocks_to_fill)
    if (block_idx >= blocks_to_fill) {
        return;
    }
    RowSlice row(trace + block_idx * NUM_ROUNDS, height);
    __align__(16) uint64_t current_state[5][5] = {0};
    __align__(16) uint64_t initial_state[5][5] = {0};

    if (block_idx < total_num_blocks) {
        // We need to transpose state matrices due to a plonky3 issue: https://github.com/Plonky3/Plonky3/issues/672
        // Note: the fix for this issue will be a commit after the major Field crate refactor PR https://github.com/Plonky3/Plonky3/pull/640
        //       which will require a significant refactor to switch to.
#pragma unroll 5
        for (auto x = 0; x < 5; x++) {
#pragma unroll 5
            for (auto y = 0; y < 5; y++) {
                current_state[x][y] = states[block_idx * KECCAK_STATE_SIZE + y + 5 * x];
                initial_state[x][y] = current_state[x][y];
            }
        }
    }
    uint16_t *initial_state_limbs = reinterpret_cast<uint16_t *>(initial_state);

    COL_WRITE_ARRAY(row, KeccakPermCols, a, initial_state_limbs);
    COL_WRITE_ARRAY(row, KeccakPermCols, preimage, initial_state_limbs);
    generate_trace_row_for_round(row, 0, current_state);

    auto last_rounds = height - (blocks_to_fill - 1) * NUM_ROUNDS;
    RowSlice prev_round_row = row;
#pragma unroll 8
    for (auto round = 1; round < NUM_ROUNDS; round++) {
        if ((block_idx == (blocks_to_fill - 1)) && (round >= last_rounds)) {
            break;
        }
        RowSlice round_row(prev_round_row.ptr + 1, height);
        COL_WRITE_ARRAY(round_row, KeccakPermCols, preimage, initial_state_limbs);
        for (auto y = 0; y < 5; y++) {
#pragma unroll 5
            for (auto x = 0; x < 5; x++) {
#pragma unroll 4
                for (auto limb = 0; limb < U64_LIMBS; limb++) {
                    // Copy previous row's output to next row's input.
                    COL_WRITE_VALUE(
                        round_row,
                        KeccakPermCols,
                        a[y][x][limb],
                        ((x == 0) && (y == 0))
                            ? prev_round_row
                                  [COL_INDEX(KeccakPermCols, a_prime_prime_prime_0_0_limbs[limb])]
                            : prev_round_row[COL_INDEX(KeccakPermCols, a_prime_prime[y][x][limb])]
                    );
                }
            }
        }
        generate_trace_row_for_round(round_row, round, current_state);
        prev_round_row = round_row;
    }
}

#define KECCAK_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, KeccakVmCols, FIELD, VALUE)
#define KECCAK_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, KeccakVmCols, FIELD, VALUES)
#define KECCAK_FILL_ZERO(FIELD) COL_FILL_ZERO(row, KeccakVmCols, FIELD)
#define KECCAK_SLICE(FIELD) row.slice_from(COL_INDEX(KeccakVmCols, FIELD))

__global__ void keccak256_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    uint32_t *block_offsets,
    uint32_t *block_to_record_idx,
    uint32_t total_num_blocks,
    uint64_t *states,
    size_t rows_used,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    auto row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto round_idx = row_idx % NUM_ROUNDS;

    RowSlice row(trace + row_idx, height);
    if (row_idx < rows_used) {
        MemoryAuxColsFactory mem_helper(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        auto bitwise_lookup = BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits);

        uint32_t global_block_idx = row_idx / NUM_ROUNDS;
        uint32_t record_idx = block_to_record_idx[global_block_idx];

        auto record_mut = KeccakVmRecordMut(records + record_offsets[record_idx]);
        KeccakVmRecordHeader vm_record = record_mut.header;
        auto len = vm_record.len;
        auto num_blocks = num_keccak_f(len);
        auto state = states + global_block_idx * KECCAK_STATE_SIZE;

        auto input = record_mut.input_ptr;

        auto block_idx = global_block_idx - block_offsets[record_idx];
        auto input_offset = block_idx * KECCAK_RATE_BYTES;
        auto start_timestamp =
            vm_record.timestamp + (block_idx * (KECCAK_REGISTER_READS + KECCAK_ABSORB_READS));
        auto rem_len = len - input_offset;
        KECCAK_WRITE(inner._export, Fp::zero());
        { // Fill the sponge columns
            KECCAK_WRITE(sponge.is_new_start, (block_idx == 0) && (round_idx == 0));
#pragma unroll 8
            for (auto i = 0; i < KECCAK_RATE_BYTES; i++) {
                KECCAK_WRITE(sponge.is_padding_byte[i], i >= rem_len);
            }

            if (block_idx == num_blocks - 1) {
                auto this_block_len = len - ((num_blocks - 1) * KECCAK_RATE_BYTES);
                for (auto i = 0; i < this_block_len; i++) {
                    KECCAK_WRITE(sponge.block_bytes[i], input[input_offset + i]);
                }
                // Add Keccak spec padding
                if (this_block_len == KECCAK_RATE_BYTES - 1) {
                    KECCAK_WRITE(sponge.block_bytes[this_block_len], 0x81);
                } else {
                    KECCAK_WRITE(sponge.block_bytes[this_block_len], 0x01);
                    for (auto i = this_block_len + 1; i < KECCAK_RATE_BYTES - 1; i++) {
                        KECCAK_WRITE(sponge.block_bytes[i], 0);
                    }
                    KECCAK_WRITE(sponge.block_bytes[KECCAK_RATE_BYTES - 1], 0x80);
                }
            } else {
                KECCAK_WRITE_ARRAY(sponge.block_bytes, input + input_offset);
            }
            if (round_idx == 0) {
#pragma unroll 8
                for (auto i = 0; i < KECCAK_RATE_U16S; i++) {
                    KECCAK_WRITE(
                        sponge.state_hi[i],
                        static_cast<uint8_t>(state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8))
                    );
                }
            } else if (round_idx == NUM_ROUNDS - 1) {
                auto next_state = state + total_num_blocks * KECCAK_STATE_SIZE;
#pragma unroll 8
                for (auto i = 0; i < KECCAK_RATE_U16S; i++) {
                    KECCAK_WRITE(
                        sponge.state_hi[i],
                        static_cast<uint8_t>(
                            next_state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8)
                        )
                    );
                }
                if (block_idx == num_blocks - 1) {
                    KECCAK_WRITE(inner._export, Fp::one());
                    auto next_state_bytes = reinterpret_cast<uint8_t *>(next_state);
#pragma unroll 8
                    for (auto i = 0; i < NUM_ABSORB_ROUNDS * sizeof(uint64_t); i++) {
                        bitwise_lookup.add_xor(0, next_state_bytes[i]);
                    }
                }
            } else {
                KECCAK_FILL_ZERO(sponge.state_hi);
            }
        }
        { // Fill the instruction columns
            KECCAK_WRITE(instruction.pc, vm_record.from_pc);
            KECCAK_WRITE(instruction.is_enabled, 1);
            KECCAK_WRITE(instruction.is_enabled_first_round, round_idx == 0);
            KECCAK_WRITE(instruction.start_timestamp, start_timestamp);
            KECCAK_WRITE(instruction.dst_ptr, vm_record.rd_ptr);
            KECCAK_WRITE(instruction.src_ptr, vm_record.rs1_ptr);
            KECCAK_WRITE(instruction.len_ptr, vm_record.rs2_ptr);
            auto vm_record_dst_limbs = reinterpret_cast<uint8_t *>(&vm_record.dst);
            KECCAK_WRITE_ARRAY(instruction.dst, vm_record_dst_limbs);
            uint32_t src = vm_record.src + (block_idx * KECCAK_RATE_BYTES);
            KECCAK_WRITE(instruction.src, src);
            auto src_limbs = reinterpret_cast<uint8_t *>(&src);
            KECCAK_WRITE_ARRAY(instruction.src_limbs, src_limbs + 1);
            auto rem_len_limbs = reinterpret_cast<uint8_t *>(&rem_len);
            KECCAK_WRITE_ARRAY(instruction.len_limbs, rem_len_limbs + 1);
            KECCAK_WRITE(instruction.remaining_len, rem_len);
        }
        { // Fill the register reads
            if (round_idx == 0 && block_idx == 0) {
                for (auto i = 0; i < KECCAK_REGISTER_READS; i++) {
                    mem_helper.fill(
                        KECCAK_SLICE(mem_oc.register_aux[i].base),
                        vm_record.register_reads_aux[i].prev_timestamp,
                        start_timestamp + i
                    );
                }

                auto msl_rshift = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
                auto msl_lshift = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - ptr_max_bits;

                bitwise_lookup.add_range(
                    (vm_record.dst >> msl_rshift) << msl_lshift,
                    (vm_record.src >> msl_rshift) << msl_lshift
                );
                bitwise_lookup.add_range(
                    (vm_record.len >> msl_rshift) << msl_lshift,
                    (vm_record.len >> msl_rshift) << msl_lshift
                );
            } else {
                KECCAK_FILL_ZERO(mem_oc.register_aux);
            }
        }
        { // Fill the absorb reads
            if (round_idx == 0) {
                auto reads_offs = block_idx * KECCAK_ABSORB_READS;
                auto num_reads =
                    min(d_div_ceil(size_t(rem_len), KECCAK_WORD_SIZE), KECCAK_ABSORB_READS);
                start_timestamp += KECCAK_REGISTER_READS;
                for (auto i = 0; i < num_reads; i++) {
                    mem_helper.fill(
                        KECCAK_SLICE(mem_oc.absorb_reads[i].base),
                        record_mut.read_aux_ptr[i + reads_offs].prev_timestamp,
                        start_timestamp + i
                    );
                }
                for (auto i = num_reads; i < KECCAK_ABSORB_READS; i++) {
                    // Zero the whole MemoryReadAuxCols struct (prev_timestamp + lt_decomp)
                    KECCAK_FILL_ZERO(mem_oc.absorb_reads[i]);
                }
            } else {
                KECCAK_FILL_ZERO(mem_oc.absorb_reads);
            }
        }
        { // Fill the digest writes
            if (block_idx == num_blocks - 1 && round_idx == NUM_ROUNDS - 1) {
                auto timestamp = start_timestamp + (KECCAK_ABSORB_READS + KECCAK_REGISTER_READS);
                for (auto i = 0; i < KECCAK_DIGEST_WRITES; i++) {
                    KECCAK_WRITE_ARRAY(
                        mem_oc.digest_writes[i].prev_data, vm_record.write_aux[i].prev_data
                    );
                    mem_helper.fill(
                        KECCAK_SLICE(mem_oc.digest_writes[i].base),
                        vm_record.write_aux[i].prev_timestamp,
                        timestamp + i
                    );
                }
            } else {
                KECCAK_FILL_ZERO(mem_oc.digest_writes);
            }
        }
        { // Fill the partial block
            auto read_len = record_mut.read_len();
            if ((block_idx == num_blocks - 1) && (read_len != len)) {
                KECCAK_WRITE_ARRAY(mem_oc.partial_block, input + read_len - KECCAK_WORD_SIZE + 1);
            } else {
                KECCAK_FILL_ZERO(mem_oc.partial_block);
            }
        }
    } else { // height > idx >= rows_used
        // inner rows are filled in p3_tracegen kernel. Ensure export flag is 0
        KECCAK_WRITE(inner._export, Fp::zero());
        // leave other permutation columns untouched
        row.fill_zero(
            COL_INDEX(KeccakVmCols, sponge),
            NUM_KECCAK_SPONGE_COLS + NUM_KECCAK_INSTRUCTION_COLS + NUM_KECCAK_MEMORY_COLS
        );
        KECCAK_WRITE(sponge.is_new_start, round_idx == 0);
        KECCAK_WRITE(sponge.block_bytes[0], Fp::one());
        KECCAK_WRITE(sponge.block_bytes[KECCAK_RATE_BYTES - 1], Fp(0x80));
        for (auto i = 0; i < KECCAK_RATE_BYTES; i++) {
            KECCAK_WRITE(sponge.is_padding_byte[i], Fp::one());
        }
    }
}

#undef KECCAK_WRITE
#undef KECCAK_WRITE_ARRAY
#undef KECCAK_FILL_ZERO
#undef KECCAK_SLICE

extern "C" int _keccak256_p3_tracegen(
    Fp *trace,
    size_t height,
    uint32_t total_num_blocks,
    uint64_t *states
) {
    auto threads = div_ceil(height, NUM_ROUNDS);
    auto [grid, block] = kernel_launch_params(threads, 256);
    p3_inner_tracegen<<<grid, block>>>(trace, height, total_num_blocks, threads, states);
    return CHECK_KERNEL();
}

extern "C" int _keccak256_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t num_records,
    size_t *offsets,
    uint32_t *block_offsets,
    uint32_t *block_to_record_idx,
    uint32_t total_num_blocks,
    uint64_t *states,
    size_t rows_used,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height > total_num_blocks * NUM_ROUNDS);
    auto [grid, block] = kernel_launch_params(height, 256);
    keccak256_tracegen<<<grid, block>>>(
        trace,
        height,
        records,
        num_records,
        offsets,
        block_offsets,
        block_to_record_idx,
        total_num_blocks,
        states,
        rows_used,
        ptr_max_bits,
        range_checker_ptr,
        range_checker_num_bins,
        bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
