use openvm_circuit::arch::StaticProgramError;
use openvm_instructions::{instruction::Instruction, riscv::RV32_MEMORY_AS};
use openvm_stark_backend::p3_field::PrimeField32;
use super::KeccakfVmExecutor;
use openvm_instructions::riscv::{RV32_REGISTER_AS};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use std::borrow::{Borrow, BorrowMut};
use std::convert::TryInto;
use std::mem::size_of;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::program::DEFAULT_PC_STEP;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct KeccakfPreCompute {
    a: u8,
}

impl KeccakfVmExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32, 
        inst: &Instruction<F>,
        data: &mut KeccakfPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode: _,
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

        *data = KeccakfPreCompute {
            a: a.as_canonical_u32() as u8,
        };

        Ok(())

    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for KeccakfVmExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<KeccakfPreCompute>()
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
        let data: &mut KeccakfPreCompute = data.borrow_mut();
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
        todo!()
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for KeccakfVmExecutor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for KeccakfVmExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        todo!()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        _chip_idx: usize,
        _pc: u32,
        _inst: &Instruction<F>,
        _data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        todo!()
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        _chip_idx: usize,
        _pc: u32,
        _inst: &Instruction<F>,
        _data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        todo!()
    }
}


#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &KeccakfPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<KeccakfPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, exec_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_E1: bool>(
    pre_compute: &KeccakfPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    // todo: implement this

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}


#[create_handler]
#[inline(always)]
#[allow(dead_code)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<KeccakfPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<KeccakfPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
}