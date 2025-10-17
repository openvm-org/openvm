use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, NativeJalOpcode, NativeRangeCheckOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

use super::JalRangeCheckExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalPreCompute<F> {
    a: u32,
    b: F,
    return_pc: F,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct RangeCheckPreCompute {
    a: u32,
    b: u8,
    c: u8,
}

impl JalRangeCheckExecutor {
    #[inline(always)]
    fn pre_compute_jal_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        jal_data: &mut JalPreCompute<F>,
    ) -> Result<(), StaticProgramError> {
        let &Instruction { opcode, a, b, .. } = inst;

        if opcode != NativeJalOpcode::JAL.global_opcode() {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let return_pc = F::from_canonical_u32(pc.wrapping_add(DEFAULT_PC_STEP));

        *jal_data = JalPreCompute { a, b, return_pc };
        Ok(())
    }

    #[inline(always)]
    fn pre_compute_range_check_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        range_check_data: &mut RangeCheckPreCompute,
    ) -> Result<(), StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, ..
        } = inst;

        if opcode != NativeRangeCheckOpcode::RANGE_CHECK.global_opcode() {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        if b > 16 || c > 14 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *range_check_data = RangeCheckPreCompute {
            a,
            b: b as u8,
            c: c as u8,
        };
        Ok(())
    }
}

impl<F> Executor<F> for JalRangeCheckExecutor
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::cmp::max(
            size_of::<JalPreCompute<F>>(),
            size_of::<RangeCheckPreCompute>(),
        )
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let &Instruction { opcode, .. } = inst;

        let is_jal = opcode == NativeJalOpcode::JAL.global_opcode();

        if is_jal {
            let jal_data: &mut JalPreCompute<F> = data.borrow_mut();
            self.pre_compute_jal_impl(pc, inst, jal_data)?;
            Ok(execute_jal_e1_handler)
        } else {
            let range_check_data: &mut RangeCheckPreCompute = data.borrow_mut();
            self.pre_compute_range_check_impl(pc, inst, range_check_data)?;
            Ok(execute_range_check_e1_handler)
        }
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let &Instruction { opcode, .. } = inst;

        let is_jal = opcode == NativeJalOpcode::JAL.global_opcode();

        if is_jal {
            let jal_data: &mut JalPreCompute<F> = data.borrow_mut();
            self.pre_compute_jal_impl(pc, inst, jal_data)?;
            Ok(execute_jal_e1_handler)
        } else {
            let range_check_data: &mut RangeCheckPreCompute = data.borrow_mut();
            self.pre_compute_range_check_impl(pc, inst, range_check_data)?;
            Ok(execute_range_check_e1_handler)
        }
    }
}

impl<F> MeteredExecutor<F> for JalRangeCheckExecutor
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::cmp::max(
            size_of::<E2PreCompute<JalPreCompute<F>>>(),
            size_of::<E2PreCompute<RangeCheckPreCompute>>(),
        )
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let &Instruction { opcode, .. } = inst;

        let is_jal = opcode == NativeJalOpcode::JAL.global_opcode();

        if is_jal {
            let pre_compute: &mut E2PreCompute<JalPreCompute<F>> = data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_jal_impl(pc, inst, &mut pre_compute.data)?;
            Ok(execute_jal_e2_handler)
        } else {
            let pre_compute: &mut E2PreCompute<RangeCheckPreCompute> = data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_range_check_impl(pc, inst, &mut pre_compute.data)?;
            Ok(execute_range_check_e2_handler)
        }
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let &Instruction { opcode, .. } = inst;

        let is_jal = opcode == NativeJalOpcode::JAL.global_opcode();

        if is_jal {
            let pre_compute: &mut E2PreCompute<JalPreCompute<F>> = data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_jal_impl(pc, inst, &mut pre_compute.data)?;
            Ok(execute_jal_e2_handler)
        } else {
            let pre_compute: &mut E2PreCompute<RangeCheckPreCompute> = data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_range_check_impl(pc, inst, &mut pre_compute.data)?;
            Ok(execute_range_check_e2_handler)
        }
    }
}

#[inline(always)]
unsafe fn execute_jal_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &JalPreCompute<F>,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    exec_state.vm_write(AS::Native as u32, pre_compute.a, &[pre_compute.return_pc]);
    // TODO(ayush): better way to do this
    *pc = (F::from_canonical_u32(*pc) + pre_compute.b).as_canonical_u32();
    *instret += 1;
}

#[inline(always)]
unsafe fn execute_range_check_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &RangeCheckPreCompute,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let [a_val]: [F; 1] = exec_state.host_read(AS::Native as u32, pre_compute.a);

    exec_state.vm_write(AS::Native as u32, pre_compute.a, &[a_val]);
    {
        let a_val = a_val.as_canonical_u32();
        let b = pre_compute.b;
        let c = pre_compute.c;
        let x = a_val & 0xffff;
        let y = a_val >> 16;

        // The range of `b`,`c` had already been checked in `pre_compute_e1`.
        if !(x < (1 << b) && y < (1 << c)) {
            let err = ExecutionError::Fail {
                pc: *pc,
                msg: "NativeRangeCheck",
            };
            return Err(err);
        }
    }
    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_jal_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &JalPreCompute<F> = pre_compute.borrow();
    execute_jal_e12_impl(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_jal_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalPreCompute<F>> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_jal_e12_impl(&pre_compute.data, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_range_check_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &RangeCheckPreCompute = pre_compute.borrow();
    execute_range_check_e12_impl(pre_compute, instret, pc, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_range_check_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<RangeCheckPreCompute> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_range_check_e12_impl(&pre_compute.data, instret, pc, exec_state)
}
