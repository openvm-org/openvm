use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_sha2_transpiler::Rv32Sha2Opcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{Sha2Config, Sha2VmExecutor, SHA2_READ_SIZE};
use crate::SHA2_WRITE_SIZE;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct Sha2PreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32, C: Sha2Config> Executor<F> for Sha2VmExecutor<C> {
    fn pre_compute_size(&self) -> usize {
        size_of::<Sha2PreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut Sha2PreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _, C>)
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
        let data: &mut Sha2PreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<_, _>)
    }
}

impl<F: PrimeField32, C: Sha2Config> MeteredExecutor<F> for Sha2VmExecutor<C> {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Sha2PreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
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
        let data: &mut E2PreCompute<Sha2PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _, C>)
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
        let data: &mut E2PreCompute<Sha2PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler::<_, _>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    C: Sha2Config,
    CTX: ExecutionCtxTrait,
    const IS_E1: bool,
>(
    pre_compute: &Sha2PreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let dst = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let state = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let input = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let dst_u32 = u32::from_le_bytes(dst);
    let state_u32 = u32::from_le_bytes(state);
    let input_u32 = u32::from_le_bytes(input);

    let mut state_data = Vec::with_capacity(C::STATE_BYTES);
    let mut input_block = Vec::with_capacity(C::BLOCK_BYTES);
    for i in 0..C::STATE_READS {
        state_data.extend_from_slice(&exec_state.vm_read::<u8, SHA2_READ_SIZE>(
            RV32_MEMORY_AS,
            state_u32 + (i * SHA2_READ_SIZE) as u32,
        ));
    }
    for i in 0..C::BLOCK_READS {
        input_block.extend_from_slice(&exec_state.vm_read::<u8, SHA2_READ_SIZE>(
            RV32_MEMORY_AS,
            input_u32 + (i * SHA2_READ_SIZE) as u32,
        ));
    }

    C::compress(&mut state_data, &input_block);

    for i in 0..C::DIGEST_WRITES {
        exec_state.vm_write::<u8, SHA2_WRITE_SIZE>(
            RV32_MEMORY_AS,
            dst_u32 + (i * SHA2_WRITE_SIZE) as u32,
            &state_data[i * SHA2_WRITE_SIZE..(i + 1) * SHA2_WRITE_SIZE]
                .try_into()
                .unwrap(),
        );
    }

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;

    1 // height delta
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, C: Sha2Config>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Sha2PreCompute = pre_compute.borrow();
    execute_e12_impl::<F, C, CTX, true>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, C: Sha2Config>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Sha2PreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, C, CTX, false>(&pre_compute.data, instret, pc, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

impl<C: Sha2Config> Sha2VmExecutor<C> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Sha2PreCompute,
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
        *data = Sha2PreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        assert_eq!(&Rv32Sha2Opcode::SHA256.global_opcode(), opcode);
        Ok(())
    }
}
