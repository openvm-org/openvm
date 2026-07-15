use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::AddSubExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct AddSubPreCompute {
    rs2_ptr: u8,
    rd_ptr: u8,
    rs1_ptr: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> AddSubExecutor<A, NUM_LIMBS, LIMB_BITS> {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut AddSubPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        if (d.as_canonical_u32() != RV64_REGISTER_AS) || (e.as_canonical_u32() != RV64_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = AddSubPreCompute {
            rs2_ptr: c.as_canonical_u32() as u8,
            rd_ptr: a.as_canonical_u32() as u8,
            rs1_ptr: b.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $opcode:expr, $offset:expr) => {
        Ok(
            match BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)) {
                BaseAluOpcode::ADD => $execute_impl::<_, AddOp>,
                BaseAluOpcode::SUB => $execute_impl::<_, SubOp>,
                _ => unreachable!("AddSubExecutor received non-ADD/SUB opcode"),
            },
        )
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for AddSubExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<AddSubPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut AddSubPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, inst.opcode, self.offset)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut AddSubPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, inst.opcode, self.offset)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for AddSubExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<AddSubPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<AddSubPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, inst.opcode, self.offset)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<AddSubPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, inst.opcode, self.offset)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: &AddSubPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 = exec_state
        .vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.rs1_ptr as u32);
    let rs2 = exec_state
        .vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.rs2_ptr as u32);
    let rs1 = u64::from_le_bytes(rs1);
    let rs2 = u64::from_le_bytes(rs2);
    let rd = <OP as AluOp>::compute(rs1, rs2);
    let rd = rd.to_le_bytes();
    exec_state.vm_write_bytes::<RV64_REGISTER_NUM_LIMBS>(
        RV64_REGISTER_AS,
        pre_compute.rd_ptr as u32,
        &rd,
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &AddSubPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<AddSubPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<AddSubPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<AddSubPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
}

trait AluOp {
    fn compute(rs1: u64, rs2: u64) -> u64;
}
struct AddOp;
struct SubOp;
impl AluOp for AddOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_add(rs2)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_sub(rs2)
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{riscv::RV64_IMM_AS, LocalOpcode};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;
    use crate::{adapters::Rv64BaseAluRegU16AdapterExecutor, Rv64AddSubExecutor};

    #[test]
    fn add_sub_reject_immediate_operand() {
        let executor = Rv64AddSubExecutor::new(
            Rv64BaseAluRegU16AdapterExecutor,
            BaseAluOpcode::CLASS_OFFSET,
        );

        for opcode in [BaseAluOpcode::ADD, BaseAluOpcode::SUB] {
            let instruction = Instruction::<BabyBear>::from_usize(
                opcode.global_opcode(),
                [
                    RV64_REGISTER_NUM_LIMBS,
                    2 * RV64_REGISTER_NUM_LIMBS,
                    1,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                ],
            );
            let mut data = AddSubPreCompute {
                rs2_ptr: 0,
                rd_ptr: 0,
                rs1_ptr: 0,
            };

            assert!(matches!(
                executor.pre_compute_impl(0, &instruction, &mut data),
                Err(StaticProgramError::InvalidInstruction(0))
            ));
        }
    }
}
