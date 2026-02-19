use std::{
    borrow::{Borrow, BorrowMut},
    slice::from_raw_parts,
};

use itertools::Itertools;
use openvm_circuit::{
    arch::{hasher::Hasher, *},
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode, NATIVE_AS,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use super::DeferralCallExecutor;
use crate::{
    poseidon2::deferral_poseidon2_chip,
    utils::{byte_commit_to_f, combine_output, COMMIT_NUM_BYTES},
    DeferralFn, CALL_AIR_IDX, POSEIDON2_AIR_IDX,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DeferralCallPrecompute<'a> {
    rd_ptr: u32,
    rs_ptr: u32,
    deferral_idx: u32,
    input_acc_ptr: u32,
    output_acc_ptr: u32,
    deferral_fn: &'a DeferralFn,
}

impl DeferralCallExecutor {
    #[inline(always)]
    fn pre_compute_impl<'a, F: VmField>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DeferralCallPrecompute<'a>,
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

        if opcode.local_opcode_idx(DeferralOpcode::CLASS_OFFSET) != DeferralOpcode::CALL as usize
            || d.as_canonical_u32() != RV32_REGISTER_AS
            || e.as_canonical_u32() != RV32_MEMORY_AS
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let deferral_idx = c.as_canonical_u32();
        let deferral_fn = self
            .deferral_fns
            .get(deferral_idx as usize)
            .ok_or(StaticProgramError::InvalidInstruction(pc))?;

        let input_acc_ptr =
            self.adapter.native_start_ptr + (2 * deferral_idx + 1) * (DIGEST_SIZE as u32);
        *data = DeferralCallPrecompute {
            rd_ptr: a.as_canonical_u32(),
            rs_ptr: b.as_canonical_u32(),
            deferral_idx,
            input_acc_ptr,
            output_acc_ptr: input_acc_ptr + (DIGEST_SIZE as u32),
            deferral_fn: deferral_fn.as_ref(),
        };

        Ok(())
    }
}

impl<F: VmField> InterpreterExecutor<F> for DeferralCallExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<DeferralCallPrecompute>()
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
        let pre_compute: &mut DeferralCallPrecompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
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
        let pre_compute: &mut DeferralCallPrecompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler::<_, _>)
    }
}

#[cfg(feature = "aot")]
impl<F: VmField> AotExecutor<F> for DeferralCallExecutor {}

impl<F: VmField> InterpreterMeteredExecutor<F> for DeferralCallExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DeferralCallPrecompute>>()
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
        let pre_compute: &mut E2PreCompute<DeferralCallPrecompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_impl::<_, _>)
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
        let pre_compute: &mut E2PreCompute<DeferralCallPrecompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler::<_, _>)
    }
}

#[cfg(feature = "aot")]
impl<F: VmField> AotMeteredExecutor<F> for DeferralCallExecutor {}

#[inline(always)]
unsafe fn execute_e12_impl<F: VmField, CTX: ExecutionCtxTrait>(
    pre_compute: &DeferralCallPrecompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let output_ptr = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.rd_ptr));
    let input_ptr = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.rs_ptr));

    let input_commit_bytes: [_; COMMIT_NUM_BYTES] = exec_state.vm_read(RV32_MEMORY_AS, input_ptr);
    let input_commit: [F; _] = byte_commit_to_f(&input_commit_bytes.map(F::from_u8));
    let old_input_acc = exec_state.vm_read(NATIVE_AS, pre_compute.input_acc_ptr);
    let old_output_acc = exec_state.vm_read(NATIVE_AS, pre_compute.output_acc_ptr);

    let poseidon2_chip = deferral_poseidon2_chip();
    let (output_commit, output_len) = pre_compute.deferral_fn.execute(
        &input_commit_bytes.to_vec(),
        &mut exec_state.streams.deferrals[pre_compute.deferral_idx as usize],
        pre_compute.deferral_idx,
        &poseidon2_chip,
    );
    let output_f_commit =
        byte_commit_to_f(&output_commit.iter().map(|v| F::from_u8(*v)).collect_vec());
    let output_key = combine_output(output_commit, output_len.to_le_bytes());

    let new_input_acc = poseidon2_chip.compress(&old_input_acc, &input_commit);
    let new_output_acc = poseidon2_chip.compress(&old_output_acc, &output_f_commit);

    exec_state.vm_write(RV32_MEMORY_AS, output_ptr, &output_key);
    exec_state.vm_write(NATIVE_AS, pre_compute.input_acc_ptr, &new_input_acc);
    exec_state.vm_write(NATIVE_AS, pre_compute.output_acc_ptr, &new_output_acc);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: VmField, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &DeferralCallPrecompute =
        from_raw_parts(pre_compute, size_of::<DeferralCallPrecompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: VmField, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<DeferralCallPrecompute> = from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<DeferralCallPrecompute>>(),
    )
    .borrow();

    execute_e12_impl(&pre_compute.data, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);

    // The Poseidon2 peripheral chip's height also increases as a result of
    // this opcode's execution. In DEFER_CALL, both the input and output
    // hash accumulator for some deferral circuit are updated.
    exec_state.ctx.on_height_change(
        pre_compute.chip_idx as usize + (POSEIDON2_AIR_IDX - CALL_AIR_IDX),
        2,
    );
}
