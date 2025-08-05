//! Stateful keccak256 hasher. Handles full keccak sponge (padding, absorb, keccak-f) on
//! variable length inputs read from VM memory.

use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_stark_backend::p3_field::PrimeField32;
use p3_keccak_air::NUM_ROUNDS;

pub mod air;
pub mod columns;
pub mod trace;
pub mod utils;

mod extension;
#[cfg(test)]
mod tests;
pub use air::KeccakVmAir;
pub use extension::*;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::Rv32KeccakOpcode;

use crate::utils::{keccak256, num_keccak_f};

// ==== Constants for register/memory adapter ====
/// Register reads to get dst, src, len
const KECCAK_REGISTER_READS: usize = 3;
/// Number of cells to read/write in a single memory access
const KECCAK_WORD_SIZE: usize = 4;
/// Memory reads for absorb per row
const KECCAK_ABSORB_READS: usize = KECCAK_RATE_BYTES / KECCAK_WORD_SIZE;
/// Memory writes for digest per row
const KECCAK_DIGEST_WRITES: usize = KECCAK_DIGEST_BYTES / KECCAK_WORD_SIZE;

// ==== Do not change these constants! ====
/// Total number of sponge bytes: number of rate bytes + number of capacity
/// bytes.
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Total number of 16-bit limbs in the sponge.
pub const KECCAK_WIDTH_U16S: usize = KECCAK_WIDTH_BYTES / 2;
/// Number of rate bytes.
pub const KECCAK_RATE_BYTES: usize = 136;
/// Number of 16-bit rate limbs.
pub const KECCAK_RATE_U16S: usize = KECCAK_RATE_BYTES / 2;
/// Number of absorb rounds, equal to rate in u64s.
pub const NUM_ABSORB_ROUNDS: usize = KECCAK_RATE_BYTES / 8;
/// Number of capacity bytes.
pub const KECCAK_CAPACITY_BYTES: usize = 64;
/// Number of 16-bit capacity limbs.
pub const KECCAK_CAPACITY_U16S: usize = KECCAK_CAPACITY_BYTES / 2;
/// Number of output digest bytes used during the squeezing phase.
pub const KECCAK_DIGEST_BYTES: usize = 32;
/// Number of 64-bit digest limbs.
pub const KECCAK_DIGEST_U64S: usize = KECCAK_DIGEST_BYTES / 8;

pub type KeccakVmChip<F> = VmChipWrapper<F, KeccakVmFiller>;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakVmStep {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct KeccakVmFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct KeccakPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> InsExecutorE1<F> for KeccakVmStep {
    fn pre_compute_size(&self) -> usize {
        size_of::<KeccakPreCompute>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E1ExecutionCtx,
    {
        let data: &mut KeccakPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
    }
}

impl<F: PrimeField32> InsExecutorE2<F> for KeccakVmStep {
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<KeccakPreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: E2ExecutionCtx,
    {
        let data: &mut E2PreCompute<KeccakPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_E1: bool>(
    pre_compute: &KeccakPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let dst = vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let src = vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let len = vm_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let dst_u32 = u32::from_le_bytes(dst);
    let src_u32 = u32::from_le_bytes(src);
    let len_u32 = u32::from_le_bytes(len);

    let (output, height) = if IS_E1 {
        // SAFETY: RV32_MEMORY_AS is memory address space of type u8
        let message = vm_state.vm_read_slice(RV32_MEMORY_AS, src_u32, len_u32 as usize);
        let output = keccak256(message);
        (output, 0)
    } else {
        let num_reads = (len_u32 as usize).div_ceil(KECCAK_WORD_SIZE);
        let message: Vec<_> = (0..num_reads)
            .flat_map(|i| {
                vm_state.vm_read::<u8, KECCAK_WORD_SIZE>(
                    RV32_MEMORY_AS,
                    src_u32 + (i * KECCAK_WORD_SIZE) as u32,
                )
            })
            .collect();
        let output = keccak256(&message[..len_u32 as usize]);
        let height = (num_keccak_f(len_u32 as usize) * NUM_ROUNDS) as u32;
        (output, height)
    };
    vm_state.vm_write(RV32_MEMORY_AS, dst_u32, &output);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    height
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &KeccakPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<KeccakPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, CTX, false>(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

impl KeccakVmStep {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut KeccakPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = KeccakPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        assert_eq!(&Rv32KeccakOpcode::KECCAK256.global_opcode(), opcode);
        Ok(())
    }
}
