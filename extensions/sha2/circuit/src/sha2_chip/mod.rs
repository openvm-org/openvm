//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.
use std::{cmp::min, iter};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        MatrixRecordArena, NewVmChipWrapper, Result, StepExecutorE1, VmStateMut,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{
    memory_read_from_state, memory_write, memory_write_from_state, read_rv32_register,
    read_rv32_register_from_state,
};
use openvm_sha2_air::{Sha256Config, Sha2StepHelper, Sha2Variant, Sha384Config, Sha512Config};
use openvm_sha2_transpiler::Rv32Sha2Opcode;
use openvm_stark_backend::p3_field::PrimeField32;
use sha2::{Digest, Sha256, Sha384, Sha512};

mod air;
mod columns;
mod config;
mod trace;
mod utils;

pub use air::*;
pub use columns::*;
pub use config::*;
pub use utils::get_sha2_num_blocks;

#[cfg(test)]
mod tests;

pub type Sha2VmChip<F, C> = NewVmChipWrapper<F, Sha2VmAir<C>, Sha2VmStep<C>, MatrixRecordArena<F>>;

pub struct Sha2VmStep<C: ShaChipConfig> {
    pub inner: Sha2StepHelper<C>,
    pub padding_encoder: Encoder,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub offset: usize,
    pub pointer_max_bits: usize,
}

impl<C: ShaChipConfig> Sha2VmStep<C> {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        offset: usize,
        pointer_max_bits: usize,
    ) -> Self {
        Self {
            inner: Sha2StepHelper::<C>::new(),
            padding_encoder: Encoder::new(PaddingFlags::COUNT, 2, false),
            bitwise_lookup_chip,
            offset,
            pointer_max_bits,
        }
    }
}

impl<F: PrimeField32, C: ShaChipConfig> StepExecutorE1<F> for Sha2VmStep<C> {
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let local_opcode = opcode.local_opcode_idx(self.offset);
        debug_assert_eq!(local_opcode, C::OPCODE.local_usize());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);
        let dst = read_rv32_register(state.memory, a.as_canonical_u32());
        let src = read_rv32_register(state.memory, b.as_canonical_u32());
        let len = read_rv32_register(state.memory, c.as_canonical_u32());

        debug_assert!(src + len <= (1 << self.pointer_max_bits));
        debug_assert!(dst < (1 << self.pointer_max_bits));

        // SAFETY: RV32_MEMORY_AS is valid address space with type u8
        let message = unsafe {
            state
                .memory
                .memory
                .get_slice::<u8>((RV32_MEMORY_AS, src), len as usize)
        };

        let output = sha2_solve::<C>(message);
        match C::OPCODE {
            Rv32Sha2Opcode::SHA256 => {
                memory_write::<{ Sha256Config::WRITE_SIZE }>(
                    state.memory,
                    RV32_MEMORY_AS,
                    dst,
                    output.as_slice().try_into().unwrap(),
                );
            }
            Rv32Sha2Opcode::SHA512 => {
                for i in 0..C::NUM_WRITES {
                    memory_write::<{ Sha512Config::WRITE_SIZE }>(
                        state.memory,
                        RV32_MEMORY_AS,
                        dst + (i * Sha512Config::WRITE_SIZE) as u32,
                        output.as_slice()
                            [i * Sha512Config::WRITE_SIZE..(i + 1) * Sha512Config::WRITE_SIZE]
                            .try_into()
                            .unwrap(),
                    );
                }
            }
            Rv32Sha2Opcode::SHA384 => {
                // Pad the output with zeros to 64 bytes
                let output = output
                    .into_iter()
                    .chain(iter::repeat(0).take(16))
                    .collect::<Vec<_>>();
                for i in 0..C::NUM_WRITES {
                    memory_write::<{ Sha384Config::WRITE_SIZE }>(
                        state.memory,
                        RV32_MEMORY_AS,
                        dst + (i * Sha384Config::WRITE_SIZE) as u32,
                        output.as_slice()
                            [i * Sha384Config::WRITE_SIZE..(i + 1) * Sha384Config::WRITE_SIZE]
                            .try_into()
                            .unwrap(),
                    );
                }
            }
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        debug_assert_eq!(*opcode, C::OPCODE.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let dst = read_rv32_register_from_state(state, a.as_canonical_u32());
        let src = read_rv32_register_from_state(state, b.as_canonical_u32());
        let len = read_rv32_register_from_state(state, c.as_canonical_u32());

        let num_blocks = get_sha2_num_blocks::<C>(len) as usize;

        // we will read [num_blocks] * [C::BLOCK_CELLS] cells but only [len] cells will be used
        debug_assert!(src as usize + num_blocks * C::BLOCK_CELLS <= (1 << self.pointer_max_bits));
        debug_assert!(dst as usize + C::WRITE_SIZE <= (1 << self.pointer_max_bits));
        // We don't support messages longer than 2^29 bytes
        debug_assert!(len < SHA_MAX_MESSAGE_LEN as u32);

        let mut input = Vec::with_capacity(len as usize);
        for idx in 0..num_blocks * C::NUM_READ_ROWS {
            let read = match C::VARIANT {
                Sha2Variant::Sha256 => {
                    memory_read_from_state::<F, MeteredCtx, { Sha256Config::READ_SIZE }>(
                        state,
                        RV32_MEMORY_AS,
                        src + (idx * C::READ_SIZE) as u32,
                    )
                    .to_vec()
                }
                Sha2Variant::Sha512 => {
                    memory_read_from_state::<F, MeteredCtx, { Sha512Config::READ_SIZE }>(
                        state,
                        RV32_MEMORY_AS,
                        src + (idx * C::READ_SIZE) as u32,
                    )
                    .to_vec()
                }
                Sha2Variant::Sha384 => {
                    memory_read_from_state::<F, MeteredCtx, { Sha384Config::READ_SIZE }>(
                        state,
                        RV32_MEMORY_AS,
                        src + (idx * C::READ_SIZE) as u32,
                    )
                    .to_vec()
                }
            };
            let offset = idx * C::READ_SIZE;
            if offset < len as usize {
                let copy_len = min(len as usize - offset, C::READ_SIZE);
                input.extend_from_slice(&read[..copy_len]);
            }
        }

        let mut output = sha2_solve::<C>(&input);
        match C::OPCODE {
            Rv32Sha2Opcode::SHA256 => {
                debug_assert!(output.len() == Sha256Config::WRITE_SIZE);
                memory_write_from_state::<F, MeteredCtx, { Sha256Config::WRITE_SIZE }>(
                    state,
                    RV32_MEMORY_AS,
                    dst,
                    output.as_slice().try_into().unwrap(),
                );
            }
            Rv32Sha2Opcode::SHA512 => {
                debug_assert!(output.len() % Sha512Config::WRITE_SIZE == 0);
                output
                    .chunks_exact(Sha512Config::WRITE_SIZE)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        memory_write_from_state::<F, MeteredCtx, { Sha512Config::WRITE_SIZE }>(
                            state,
                            RV32_MEMORY_AS,
                            dst + (i * Sha512Config::WRITE_SIZE) as u32,
                            chunk.try_into().unwrap(),
                        );
                    });
            }
            Rv32Sha2Opcode::SHA384 => {
                output.extend(iter::repeat(0).take(16));
                debug_assert!(output.len() % Sha384Config::WRITE_SIZE == 0);
                output
                    .chunks_exact(Sha384Config::WRITE_SIZE)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        memory_write_from_state::<F, MeteredCtx, { Sha384Config::WRITE_SIZE }>(
                            state,
                            RV32_MEMORY_AS,
                            dst + (i * Sha384Config::WRITE_SIZE) as u32,
                            chunk.try_into().unwrap(),
                        );
                    });
            }
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        state.ctx.trace_heights[chip_index] += (num_blocks * C::ROWS_PER_BLOCK) as u32;
        Ok(())
    }
}

pub fn sha2_solve<C: ShaChipConfig>(input_message: &[u8]) -> Vec<u8> {
    match C::VARIANT {
        Sha2Variant::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
        Sha2Variant::Sha512 => {
            let mut hasher = Sha512::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
        Sha2Variant::Sha384 => {
            let mut hasher = Sha384::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
    }
}
