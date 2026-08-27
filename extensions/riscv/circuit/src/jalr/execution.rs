use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::JalrOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::{checked_jalr_target, JalrExecutor};
use crate::adapters::bytes_to_u32;
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalrPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
}

impl JalrExecutor {
    /// Return true if enabled.
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut JalrPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let imm_extended = inst.c.as_u32() + inst.g.as_u32() * 0xffff0000;
        if inst.d.as_u32() != REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = JalrPreCompute {
            imm_extended,
            a: inst.a.as_u32() as u8,
            b: inst.b.as_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok(enabled)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $enabled:ident) => {
        if $enabled {
            Ok($execute_impl::<_, true>)
        } else {
            Ok($execute_impl::<_, false>)
        }
    };
}

impl<F> InterpreterExecutor<F> for JalrExecutor
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            JalrOpcode::from_usize(opcode - JalrOpcode::CLASS_OFFSET)
        )
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalrPreCompute>()
    }
    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
    }
}

impl<F> InterpreterMeteredExecutor<F> for JalrExecutor
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalrPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: &JalrPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let rs1 = exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.b as u32);
    let rs1 = bytes_to_u32(rs1);
    let (_, to_pc) =
        checked_jalr_target(rs1, pre_compute.imm_extended).ok_or(ExecutionError::Fail {
            pc,
            msg: "JALR target is outside implemented PC address space or misaligned",
        })?;
    let rd = (u64::from(pc) + u64::from(DEFAULT_PC_STEP)).to_le_bytes();

    if ENABLED {
        exec_state.vm_write_bytes(REGISTER_AS, pre_compute.a as u32, &rd);
    } else {
        exec_state.ctx.advance_timestamp(1);
    }

    exec_state.set_pc(to_pc);
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &JalrPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<JalrPreCompute>()).borrow();
    execute_e12_impl::<CTX, ENABLED>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<JalrPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<JalrPreCompute>>()).borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, ENABLED>(&pre_compute.data, exec_state)
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::{
        execution_mode::{ExecutionCtx, MeteredCostCtx},
        InterpretedInstance, Streams, VmExecutionConfig,
    };
    use openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        program::Program,
        SystemOpcode,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;
    use crate::Rv64IConfig;

    fn jalr_exe(rs1: u32, imm: usize) -> VmExe {
        let jalr = Instruction::from_usize(
            JalrOpcode::JALR.global_opcode(),
            [16, 8, imm, REGISTER_AS as usize, 0, 1, 0],
        );
        let terminate = Instruction {
            opcode: SystemOpcode::TERMINATE.global_opcode(),
            ..Default::default()
        };
        let mut init_memory = SparseMemoryImage::new();
        for (i, byte) in u64::from(rs1).to_le_bytes().into_iter().enumerate() {
            init_memory.insert((REGISTER_AS, 8 + i as u32), byte);
        }
        VmExe::new(Program::new_without_debug_infos(
            &[jalr, terminate.clone(), terminate],
            0,
        ))
        .with_init_memory(init_memory)
    }

    fn assert_invalid_target(error: ExecutionError) {
        assert!(matches!(
            error,
            ExecutionError::Fail {
                pc: 0,
                msg: "JALR target is outside implemented PC address space or misaligned"
            }
        ));
    }

    #[test]
    fn jalr_invalid_targets_pure_or_tco_execution() {
        let config = Rv64IConfig::default();
        let inventory =
            <Rv64IConfig as VmExecutionConfig<BabyBear>>::create_executors(&config).unwrap();
        for exe in [jalr_exe(0xffff_fff8, 16), jalr_exe(0, 2)] {
            let interpreter =
                InterpretedInstance::<ExecutionCtx>::new::<BabyBear, _>(&inventory, &exe).unwrap();
            let error = interpreter
                .execute(Streams::default())
                .err()
                .expect("invalid JALR target must fail");
            assert_invalid_target(error);
        }
    }

    #[test]
    fn jalr_invalid_targets_metered_or_tco_execution() {
        let config = Rv64IConfig::default();
        let inventory =
            <Rv64IConfig as VmExecutionConfig<BabyBear>>::create_executors(&config).unwrap();
        let executor_idx_to_air_idx = vec![0; inventory.executors.len()];
        for exe in [jalr_exe(0xffff_fff8, 16), jalr_exe(0, 2)] {
            let interpreter = InterpretedInstance::<MeteredCostCtx>::new_metered::<BabyBear, _>(
                &inventory,
                &exe,
                &executor_idx_to_air_idx,
            )
            .unwrap();
            let error = interpreter
                .execute_metered_cost(Streams::default(), MeteredCostCtx::new(vec![0]))
                .err()
                .expect("invalid JALR target must fail");
            assert_invalid_target(error);
        }
    }
}
