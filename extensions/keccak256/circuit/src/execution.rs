use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use p3_keccak_air::NUM_ROUNDS;

use super::{KeccakVmExecutor, KECCAK_WORD_SIZE};
use crate::utils::{keccak256, num_keccak_f};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct KeccakPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl KeccakVmExecutor {
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

impl<F: PrimeField32> Executor<F> for KeccakVmExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<KeccakPreCompute>()
    }

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut KeccakPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut KeccakPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_tco_handler)
    }
}

impl<F: PrimeField32> MeteredExecutor<F> for KeccakVmExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<KeccakPreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<KeccakPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<KeccakPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_tco_handler::<_, _>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_E1: bool>(
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

#[create_tco_handler]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &KeccakPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<KeccakPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, CTX, false>(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}
