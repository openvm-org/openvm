use std::borrow::{Borrow, BorrowMut};

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
            Ok(execute_jal_e1_impl)
        } else {
            let range_check_data: &mut RangeCheckPreCompute = data.borrow_mut();
            self.pre_compute_range_check_impl(pc, inst, range_check_data)?;
            Ok(execute_range_check_e1_impl)
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
            Ok(execute_jal_e1_tco_handler)
        } else {
            let range_check_data: &mut RangeCheckPreCompute = data.borrow_mut();
            self.pre_compute_range_check_impl(pc, inst, range_check_data)?;
            Ok(execute_range_check_e1_tco_handler)
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
            Ok(execute_jal_e2_impl)
        } else {
            let pre_compute: &mut E2PreCompute<RangeCheckPreCompute> = data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_range_check_impl(pc, inst, &mut pre_compute.data)?;
            Ok(execute_range_check_e2_impl)
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
            Ok(execute_jal_e2_tco_handler)
        } else {
            let pre_compute: &mut E2PreCompute<RangeCheckPreCompute> = data.borrow_mut();
            pre_compute.chip_idx = chip_idx as u32;

            self.pre_compute_range_check_impl(pc, inst, &mut pre_compute.data)?;
            Ok(execute_range_check_e2_tco_handler)
        }
    }
}

#[inline(always)]
unsafe fn execute_jal_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &JalPreCompute<F>,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    vm_state.vm_write(AS::Native as u32, pre_compute.a, &[pre_compute.return_pc]);
    // TODO(ayush): better way to do this
    vm_state.pc = (F::from_canonical_u32(vm_state.pc) + pre_compute.b).as_canonical_u32();
    vm_state.instret += 1;
}

#[inline(always)]
unsafe fn execute_range_check_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &RangeCheckPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let [a_val]: [F; 1] = vm_state.host_read(AS::Native as u32, pre_compute.a);

    vm_state.vm_write(AS::Native as u32, pre_compute.a, &[a_val]);
    {
        let a_val = a_val.as_canonical_u32();
        let b = pre_compute.b;
        let c = pre_compute.c;
        let x = a_val & 0xffff;
        let y = a_val >> 16;

        // The range of `b`,`c` had already been checked in `pre_compute_e1`.
        if !(x < (1 << b) && y < (1 << c)) {
            vm_state.exit_code = Err(ExecutionError::Fail {
                pc: vm_state.pc,
                msg: "NativeRangeCheck",
            });
            return;
        }
    }
    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[create_tco_handler]
unsafe fn execute_jal_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &JalPreCompute<F> = pre_compute.borrow();
    execute_jal_e12_impl(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_jal_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalPreCompute<F>> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_jal_e12_impl(&pre_compute.data, vm_state);
}

#[create_tco_handler]
unsafe fn execute_range_check_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &RangeCheckPreCompute = pre_compute.borrow();
    execute_range_check_e12_impl(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_range_check_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<RangeCheckPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_range_check_e12_impl(&pre_compute.data, vm_state);
}
