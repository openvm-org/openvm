use std::{array::from_fn, borrow::BorrowMut, cmp::min};

use openvm_circuit::{
    arch::{Result, TraceStep, VmStateMut},
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};
use p3_keccak_air::{
    generate_trace_rows, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS, NUM_ROUNDS, U64_LIMBS,
};
use tiny_keccak::{keccakf, Hasher, Keccak};

use super::{
    columns::KeccakVmCols, KECCAK_ABSORB_READS, KECCAK_DIGEST_WRITES, KECCAK_RATE_BYTES,
    KECCAK_REGISTER_READS, NUM_ABSORB_ROUNDS,
};
use crate::{columns::NUM_KECCAK_VM_COLS, utils::num_keccak_f, KeccakVmStep, KECCAK_WORD_SIZE};

impl<F: PrimeField32, CTX> TraceStep<F, CTX> for KeccakVmStep {
    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        debug_assert_eq!(opcode, Rv32KeccakOpcode::KECCAK256.global_opcode());
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);

        let trace = &mut trace[*trace_offset..];
        let (dst, mut src, mut remaining_len) = {
            let cols: &mut KeccakVmCols<F> = trace[..width].borrow_mut();
            cols.instruction.start_timestamp = F::from_canonical_u32(state.memory.timestamp());

            let a = a.as_canonical_u32();
            let b = b.as_canonical_u32();
            let c = c.as_canonical_u32();
            let dst = tracing_read(state.memory, d, a, &mut cols.mem_oc.register_aux[0]);
            let src = tracing_read(state.memory, d, b, &mut cols.mem_oc.register_aux[1]);
            let len = tracing_read(state.memory, d, c, &mut cols.mem_oc.register_aux[2]);
            (
                dst,
                u32::from_le_bytes(src),
                u32::from_le_bytes(len) as usize,
            )
        };

        // Due to the AIR constraints, the final memory timestamp should be the following:
        let final_timestamp = state.memory.timestamp()
            + (remaining_len + KECCAK_ABSORB_READS + KECCAK_DIGEST_WRITES) as u32;
        let num_blocks = num_keccak_f(remaining_len);
        let mut hasher = Keccak::v256();

        trace
            .chunks_mut(width * NUM_ROUNDS)
            .enumerate()
            .take(num_blocks)
            .for_each(|(block_idx, chunk)| {
                let cols: &mut KeccakVmCols<F> = chunk[..NUM_KECCAK_VM_COLS].borrow_mut();
                if block_idx != 0 {
                    cols.instruction.start_timestamp =
                        F::from_canonical_u32(state.memory.timestamp());

                    state
                        .memory
                        .increment_timestamp_by(KECCAK_REGISTER_READS as u32);
                }
                cols.instruction.dst_ptr = a;
                cols.instruction.src_ptr = b;
                cols.instruction.len_ptr = c;
                cols.instruction.dst = dst.map(F::from_canonical_u8);
                cols.instruction
                    .src_limbs
                    .copy_from_slice(&src.to_le_bytes().map(F::from_canonical_u8)[1..]);
                cols.instruction.len_limbs.copy_from_slice(
                    &(remaining_len as u32)
                        .to_le_bytes()
                        .map(F::from_canonical_u8)[1..],
                );
                cols.instruction.src = F::from_canonical_u32(src);
                cols.instruction.remaining_len = F::from_canonical_usize(remaining_len);
                cols.instruction.pc = F::from_canonical_u32(*state.pc);
                cols.sponge.is_new_start = F::from_bool(block_idx == 0);

                for i in (0..KECCAK_RATE_BYTES).step_by(KECCAK_WORD_SIZE) {
                    if i < remaining_len {
                        let read = tracing_read::<_, KECCAK_WORD_SIZE>(
                            state.memory,
                            e,
                            src + i as u32,
                            &mut cols.mem_oc.absorb_reads[i / KECCAK_WORD_SIZE],
                        );
                        let copy_len = min(KECCAK_WORD_SIZE, remaining_len - i);
                        hasher.update(&read[..copy_len]);
                        cols.sponge.block_bytes[i..i + copy_len]
                            .copy_from_slice(&read.map(F::from_canonical_u8)[..copy_len]);
                        if copy_len != KECCAK_WORD_SIZE {
                            cols.mem_oc
                                .partial_block
                                .copy_from_slice(&read.map(F::from_canonical_u8)[1..]);
                        }
                    } else {
                        state.memory.increment_timestamp();
                    }
                }
                if block_idx == num_blocks - 1 {
                    if remaining_len == KECCAK_RATE_BYTES - 1 {
                        cols.sponge.block_bytes[remaining_len] = F::from_canonical_u32(0b1000_0001);
                    } else {
                        cols.sponge.block_bytes[remaining_len] = F::from_canonical_u32(0x01);
                        cols.sponge.block_bytes[KECCAK_RATE_BYTES - 1] =
                            F::from_canonical_u32(0x80);
                    }
                } else {
                    src += KECCAK_RATE_BYTES as u32;
                    remaining_len -= KECCAK_RATE_BYTES;
                }
            });

        let last_row_offset = (num_blocks * NUM_ROUNDS - 1) * width;
        let last_row: &mut KeccakVmCols<F> =
            trace[last_row_offset..last_row_offset + NUM_KECCAK_VM_COLS].borrow_mut();
        let mut digest = [0u8; 32];
        hasher.finalize(&mut digest);
        for (i, word) in digest.chunks_exact(KECCAK_WORD_SIZE).enumerate() {
            tracing_write::<_, KECCAK_WORD_SIZE>(
                state.memory,
                e,
                u32::from_le_bytes(dst) + (i * KECCAK_WORD_SIZE) as u32,
                word.try_into().unwrap(),
                &mut last_row.mem_oc.digest_writes[i],
            );
        }

        state.memory.timestamp = final_timestamp;
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        *trace_offset += num_blocks * NUM_ROUNDS * width;
        Ok(())
    }

    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut [F],
        width: usize,
        rows_used: usize,
    ) where
        Self: Send + Sync,
        F: Send + Sync,
    {
        let num_blocks = rows_used.div_ceil(NUM_ROUNDS);
        let mut states = Vec::with_capacity(num_blocks);
        let mut state = [0u64; 25];
        trace
            .chunks_mut(width * NUM_ROUNDS)
            .take(num_blocks)
            .for_each(|chunk| {
                let cols: &mut KeccakVmCols<F> = chunk[..NUM_KECCAK_VM_COLS].borrow_mut();
                if cols.sponge.is_new_start.is_one() {
                    // a new instruction is starting
                    state = [0u64; 25];
                }
                // absorb
                for (bytes, s) in cols
                    .sponge
                    .block_bytes
                    .chunks_exact(8)
                    .zip(state.iter_mut())
                {
                    // u64 <-> bytes conversion is little-endian
                    for (i, &byte) in bytes.iter().enumerate() {
                        let byte = byte.as_canonical_u32();
                        let s_byte = (*s >> (i * 8)) as u8;
                        // Update bitwise lookup (i.e. xor) chip state: order matters!
                        if cols.sponge.is_new_start.is_zero() {
                            self.bitwise_lookup_chip.request_xor(byte, s_byte as u32);
                        }
                        *s ^= (byte as u64) << (i * 8);
                    }
                }
                states.push(state);
                keccakf(&mut state);
            });

        // We need to transpose state matrices due to a plonky3 issue: https://github.com/Plonky3/Plonky3/issues/672
        // Note: the fix for this issue will be a commit after the major Field crate refactor PR https://github.com/Plonky3/Plonky3/pull/640
        //       which will require a significant refactor to switch to.
        let p3_states = states
            .par_iter()
            .map(|state| {
                // transpose of 5x5 matrix
                from_fn(|i| {
                    let x = i / 5;
                    let y = i % 5;
                    state[x + 5 * y]
                })
            })
            .collect();

        let p3_keccak_trace: RowMajorMatrix<F> = generate_trace_rows(p3_states, 0);
        trace
            .par_chunks_mut(width * NUM_ROUNDS)
            .zip(
                p3_keccak_trace
                    .values
                    .par_chunks(NUM_KECCAK_PERM_COLS * NUM_ROUNDS),
            )
            .enumerate()
            .for_each(|(block_idx, (block, p3_keccak_block))| {
                // let cols: &mut KeccakVmCols<F> = block[..NUM_KECCAK_VM_COLS].borrow_mut();
                if block_idx >= num_blocks {
                    // fill in a dummy row
                    block
                        .par_chunks_mut(width)
                        .zip(p3_keccak_block.par_chunks_exact(NUM_KECCAK_PERM_COLS))
                        .for_each(|(row, p3_keccak_row)| {
                            row[..NUM_KECCAK_PERM_COLS].copy_from_slice(p3_keccak_row);
                            let cols: &mut KeccakVmCols<F> = row.borrow_mut();
                            cols.sponge.block_bytes[0] = F::ONE;
                            cols.sponge.block_bytes[KECCAK_RATE_BYTES - 1] =
                                F::from_canonical_u32(0x80);
                            cols.sponge.is_padding_byte[0..KECCAK_RATE_BYTES].fill(F::ONE);
                        });

                    // The first row of the `dummy` block should have `is_new_start = F::ONE`
                    let first_dummy_row: &mut KeccakVmCols<F> = block[..width].borrow_mut();
                    first_dummy_row.sponge.is_new_start = F::ONE;
                    return;
                }

                // the first row is treated differently
                let (first_row, block) = block.split_at_mut(width);
                first_row[..NUM_KECCAK_PERM_COLS]
                    .copy_from_slice(&p3_keccak_block[..NUM_KECCAK_PERM_COLS]);
                let first_row: &mut KeccakVmCols<F> = first_row.borrow_mut();
                first_row.instruction.is_enabled = F::ONE;
                let remaining_len = first_row.instruction.remaining_len.as_canonical_u32() as usize;
                for i in remaining_len..KECCAK_RATE_BYTES {
                    first_row.sponge.is_padding_byte[i] = F::ONE;
                }

                for (row, p3_keccak_row) in block
                    .chunks_exact_mut(width)
                    .zip(p3_keccak_block.chunks_exact(NUM_KECCAK_PERM_COLS).skip(1))
                {
                    // Safety: `KeccakPermCols` **must** be the first field in `KeccakVmCols`
                    row[..NUM_KECCAK_PERM_COLS].copy_from_slice(p3_keccak_row);
                    let cols: &mut KeccakVmCols<F> = row.borrow_mut();

                    cols.instruction = first_row.instruction;
                    cols.sponge.block_bytes = first_row.sponge.block_bytes;
                    cols.sponge.is_padding_byte = first_row.sponge.is_padding_byte;
                    cols.mem_oc.partial_block = first_row.mem_oc.partial_block;
                }

                let (_, last_row) = block.split_at_mut(width * (NUM_ROUNDS - 2));
                let last_row: &mut KeccakVmCols<F> = last_row.borrow_mut();

                first_row.instruction.is_enabled_first_round = first_row.instruction.is_enabled;
                first_row.sponge.state_hi = from_fn(|i| {
                    F::from_canonical_u8(
                        (states[block_idx][i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8)) as u8,
                    )
                });

                let start_timestamp = first_row.instruction.start_timestamp.as_canonical_u32();
                first_row
                    .mem_oc
                    .absorb_reads
                    .par_iter_mut()
                    .take(remaining_len.div_ceil(KECCAK_WORD_SIZE))
                    .enumerate()
                    .for_each(|(i, read)| {
                        mem_helper.fill_from_prev(
                            start_timestamp + KECCAK_REGISTER_READS as u32 + i as u32,
                            read.as_mut(),
                        );
                    });

                // Check if the first row is a new start (e.g. register reads happened)
                if first_row.sponge.is_new_start.is_one() {
                    let limb_shift_bits =
                        RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;

                    self.bitwise_lookup_chip.request_range(
                        first_row.instruction.dst[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32()
                            << limb_shift_bits,
                        first_row.instruction.src_limbs[RV32_REGISTER_NUM_LIMBS - 2]
                            .as_canonical_u32()
                            << limb_shift_bits,
                    );
                    self.bitwise_lookup_chip.request_range(
                        first_row.instruction.len_limbs[RV32_REGISTER_NUM_LIMBS - 2]
                            .as_canonical_u32()
                            << limb_shift_bits,
                        first_row.instruction.len_limbs[RV32_REGISTER_NUM_LIMBS - 2]
                            .as_canonical_u32()
                            << limb_shift_bits,
                    );
                    first_row
                        .mem_oc
                        .register_aux
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, aux)| {
                            mem_helper.fill_from_prev(start_timestamp + i as u32, aux.as_mut());
                        });
                }

                let mut state = states[block_idx];
                keccakf(&mut state);
                last_row.sponge.state_hi = from_fn(|i| {
                    F::from_canonical_u8((state[i / U64_LIMBS] >> ((i % U64_LIMBS) * 16 + 8)) as u8)
                });
                last_row.inner.export = last_row.instruction.is_enabled
                    * F::from_bool(remaining_len < KECCAK_RATE_BYTES);

                // Check if this is the last block (e.g. digest write happened)
                if remaining_len < KECCAK_RATE_BYTES {
                    let write_timestamp =
                        start_timestamp + KECCAK_REGISTER_READS as u32 + KECCAK_ABSORB_READS as u32;
                    last_row
                        .mem_oc
                        .digest_writes
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, write)| {
                            mem_helper.fill_from_prev(write_timestamp + i as u32, write.as_mut());
                        });
                    for s in state.into_iter().take(NUM_ABSORB_ROUNDS) {
                        for s_byte in s.to_le_bytes() {
                            self.bitwise_lookup_chip.request_xor(0, s_byte as u32);
                        }
                    }
                }
            });
    }

    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", Rv32KeccakOpcode::KECCAK256)
    }
}
