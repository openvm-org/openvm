use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_MEMORY_AS, LocalOpcode,
};
use openvm_native_compiler::{conversion::AS, CastfOpcode};
use openvm_rv32im_circuit::adapters::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::{run_castf, CastFCoreExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct CastFPreCompute {
    a: u32,
    b: u32,
}

impl<A> CastFCoreExecutor<A> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut CastFPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            a, b, d, e, opcode, ..
        } = inst;

        if opcode.local_opcode_idx(CastfOpcode::CLASS_OFFSET) != CastfOpcode::CASTF as usize {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if d.as_canonical_u32() != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if e.as_canonical_u32() != AS::Native as u32 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        *data = CastFPreCompute { a, b };

        Ok(())
    }
}

impl<F, A> Executor<F> for CastFCoreExecutor<A>
where
    F: PrimeField32,
{
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
        let pre_compute: &mut CastFPreCompute = data.borrow_mut();

        self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = execute_e1_tco_handler::<_, _>;

        Ok(fn_ptr)
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<CastFPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut CastFPreCompute = data.borrow_mut();

        self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = execute_e1_impl::<_, _>;

        Ok(fn_ptr)
    }
}

impl<F, A> MeteredExecutor<F> for CastFCoreExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<CastFPreCompute>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<CastFPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = execute_e2_impl::<_, _>;

        Ok(fn_ptr)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<CastFPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = execute_e2_tco_handler::<_, _>;

        Ok(fn_ptr)
    }
}

#[create_tco_handler]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &CastFPreCompute = pre_compute.borrow();
    execute_e12_impl(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<CastFPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, vm_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &CastFPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let y = vm_state.vm_read::<F, 1>(AS::Native as u32, pre_compute.b)[0];
    let x = run_castf(y.as_canonical_u32());

    vm_state.vm_write::<u8, RV32_REGISTER_NUM_LIMBS>(RV32_MEMORY_AS, pre_compute.a, &x);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}
