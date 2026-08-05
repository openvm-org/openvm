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
        InterpreterMeteredExecutor, MeteredExecutionCtxTrait, StaticProgramError, VmExecState,
    },
    system::memory::online::{GuestMemory, LinearMemory},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
    PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::REVEAL_ACCESS_WIDTH;
use crate::adapters::{address_add_imm, bytes_to_u32, sign_extend_imm16};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct RevealExecutor {
    offset: usize,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct RevealPreCompute {
    imm_extended: u32,
    src_ptr: u8,
    base_ptr: u8,
}

#[inline(always)]
fn checked_reveal_address(
    pc: u32,
    base: u32,
    imm_extended: u32,
    public_values_capacity: usize,
) -> Result<u32, ExecutionError> {
    let address = address_add_imm(base, imm_extended);
    let end = address.checked_add(REVEAL_ACCESS_WIDTH as u64);
    if address > u64::from(u32::MAX) || end.is_none_or(|end| end > public_values_capacity as u64) {
        return Err(ExecutionError::Fail {
            pc,
            msg: "reveal address exceeds configured public-values capacity",
        });
    }
    Ok(address as u32)
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
        let valid_register_ptr =
            |ptr: u32| ptr <= u8::MAX as u32 && ptr.is_multiple_of(REGISTER_NUM_LIMBS as u32);
        let src_ptr = a.as_canonical_u32();
        let base_ptr = b.as_canonical_u32();
        let imm = c.as_canonical_u32();
        let imm_sign = g.as_canonical_u32();
        if opcode.local_opcode_idx(self.offset) != RevealOpcode::REVEAL as usize
            || !valid_register_ptr(src_ptr)
            || !valid_register_ptr(base_ptr)
            || imm > u16::MAX as u32
            || d.as_canonical_u32() != REGISTER_AS
            || e.as_canonical_u32() != PUBLIC_VALUES_AS
            || !f.is_one()
            || imm_sign > 1
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = RevealPreCompute {
            imm_extended: sign_extend_imm16(imm, imm_sign),
            src_ptr: src_ptr as u8,
            base_ptr: base_ptr as u8,
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
    let base_bytes: [u8; REGISTER_NUM_LIMBS] =
        exec_state.vm_read_bytes(REGISTER_AS, u32::from(pre_compute.base_ptr));
    let base = bytes_to_u32(base_bytes);
    let public_values_capacity = exec_state
        .memory
        .memory
        .get_memory()
        .get(PUBLIC_VALUES_AS as usize)
        .map(LinearMemory::size)
        .ok_or(ExecutionError::Fail {
            pc,
            msg: "public-values address space is not configured",
        })?;
    let address =
        checked_reveal_address(pc, base, pre_compute.imm_extended, public_values_capacity)?;
    let value: [u8; REVEAL_ACCESS_WIDTH] =
        exec_state.vm_read_bytes(REGISTER_AS, u32::from(pre_compute.src_ptr));
    exec_state.vm_write_bytes(PUBLIC_VALUES_AS, address, &value);
    if (address as usize).is_multiple_of(REVEAL_ACCESS_WIDTH) {
        exec_state.ctx.advance_timestamp(1);
    }
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

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::ExecutionError;

    use super::checked_reveal_address;

    #[test]
    fn reveal_address_is_bounded_by_configured_public_values_capacity() {
        assert_eq!(checked_reveal_address(4, 56, 0, 64).unwrap(), 56);
        assert!(matches!(
            checked_reveal_address(4, 57, 0, 64),
            Err(ExecutionError::Fail {
                pc: 4,
                msg: "reveal address exceeds configured public-values capacity",
            })
        ));
        assert!(checked_reveal_address(4, 0, u32::MAX, 64).is_err());
    }
}
