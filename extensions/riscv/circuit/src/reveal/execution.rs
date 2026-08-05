use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    slice::from_raw_parts,
};

#[cfg(feature = "tco")]
use openvm_circuit::arch::execution::Handler;
#[cfg(not(feature = "tco"))]
use openvm_circuit::arch::ExecuteFunc;
use openvm_circuit::{
    arch::{
        create_handler, E2PreCompute, ExecutionCtxTrait, ExecutionError, InterpreterExecutor,
        InterpreterMeteredExecutor, MeteredExecutionCtxTrait, PublicValuesStateError,
        StaticProgramError, VmExecState,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{is_valid_register_pointer, REGISTER_AS},
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[derive(Clone, Copy, derive_new::new)]
pub struct RevealExecutor {
    pub offset: usize,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct RevealPreCompute {
    src_ptr: u8,
}

impl RevealExecutor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut RevealPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;
        let src_ptr = a.as_canonical_u32();
        if opcode.local_opcode_idx(self.offset) != RevealOpcode::REVEAL as usize
            || !is_valid_register_pointer(src_ptr)
            || !b.is_zero()
            || !c.is_zero()
            || !d.is_zero()
            || !e.is_zero()
            || !f.is_zero()
            || !g.is_zero()
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = RevealPreCompute {
            src_ptr: src_ptr as u8,
        };
        Ok(())
    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for RevealExecutor {
    fn get_opcode_name(&self, _opcode: usize) -> String {
        "REVEAL".to_owned()
    }

    fn pre_compute_size(&self) -> usize {
        size_of::<RevealPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let pre_compute: &mut RevealPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler::<_>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let pre_compute: &mut RevealPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler::<_>)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for RevealExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<RevealPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<RevealPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler::<_>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<RevealPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler::<_>)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<Ctx: ExecutionCtxTrait>(
    pre_compute: &RevealPreCompute,
    exec_state: &mut VmExecState<GuestMemory, Ctx>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let value =
        u64::from_le_bytes(exec_state.vm_read_bytes(REGISTER_AS, u32::from(pre_compute.src_ptr)));
    exec_state
        .public_values
        .try_push(value)
        .map_err(|err| match err {
            PublicValuesStateError::CapacityExceeded { max_public_values } => {
                ExecutionError::PublicValuesCapacityExceeded {
                    pc,
                    max_public_values,
                }
            }
        })?;
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<Ctx: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, Ctx>,
) -> Result<(), ExecutionError> {
    let pre_compute: &RevealPreCompute =
        from_raw_parts(pre_compute, size_of::<RevealPreCompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<Ctx: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, Ctx>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<RevealPreCompute> =
        from_raw_parts(pre_compute, size_of::<E2PreCompute<RevealPreCompute>>()).borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, exec_state)
}
