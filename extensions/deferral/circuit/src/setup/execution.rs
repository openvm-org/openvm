use std::{
    borrow::{Borrow, BorrowMut},
    slice::from_raw_parts,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode, NATIVE_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use super::DeferralSetupExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DeferralSetupPrecompute<F> {
    native_start_ptr: u32,
    expected_vks_commit: [F; DIGEST_SIZE],
}

impl<F: PrimeField32> DeferralSetupExecutor<F> {
    #[inline(always)]
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DeferralSetupPrecompute<F>,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, d, opcode, .. } = inst;

        if opcode.local_opcode_idx(DeferralOpcode::CLASS_OFFSET) != DeferralOpcode::SETUP as usize {
            return Err(StaticProgramError::InvalidInstruction(pc));
        } else if a.as_canonical_u32() != self.adapter.native_start_ptr {
            return Err(StaticProgramError::InvalidInstruction(pc));
        } else if d.as_canonical_u32() != NATIVE_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        data.native_start_ptr = self.adapter.native_start_ptr;
        data.expected_vks_commit = self.expected_def_vks_commit;

        Ok(())
    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for DeferralSetupExecutor<F> {
    fn pre_compute_size(&self) -> usize {
        size_of::<DeferralSetupPrecompute<F>>()
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
        let pre_compute: &mut DeferralSetupPrecompute<F> = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        let fn_ptr = execute_e1_impl::<_, _>;
        Ok(fn_ptr)
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
        let pre_compute: &mut DeferralSetupPrecompute<F> = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        let fn_ptr = execute_e1_handler::<_, _>;
        Ok(fn_ptr)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for DeferralSetupExecutor<F> {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DeferralSetupPrecompute<F>>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<DeferralSetupPrecompute<F>> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        let fn_ptr = execute_e2_impl::<_, _>;
        Ok(fn_ptr)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<DeferralSetupPrecompute<F>> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        let fn_ptr = execute_e2_handler::<_, _>;
        Ok(fn_ptr)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for DeferralSetupExecutor<F> {}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for DeferralSetupExecutor<F> {}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &DeferralSetupPrecompute<F> =
        from_raw_parts(pre_compute, size_of::<DeferralSetupPrecompute<F>>()).borrow();
    execute_e12_impl(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<DeferralSetupPrecompute<F>> = from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<DeferralSetupPrecompute<F>>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, exec_state);
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &DeferralSetupPrecompute<F>,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    exec_state.vm_write::<F, DIGEST_SIZE>(
        NATIVE_AS,
        pre_compute.native_start_ptr,
        &pre_compute.expected_vks_commit,
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}
