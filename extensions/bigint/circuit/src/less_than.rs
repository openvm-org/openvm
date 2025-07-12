use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32LessThan256Opcode;
use openvm_circuit::arch::{
    execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
    E2PreCompute, ExecuteFunc,
    ExecutionError::InvalidInstruction,
    MatrixRecordArena, NewVmChipWrapper, StepExecutorE1, StepExecutorE2, VmAirWrapper,
    VmSegmentState,
};
use openvm_circuit_derive::{TraceFiller, TraceStep};
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::{Rv32HeapAdapterAir, Rv32HeapAdapterStep};
use openvm_rv32im_circuit::{LessThanCoreAir, LessThanStep};
use openvm_rv32im_transpiler::LessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{common, INT256_NUM_LIMBS, RV32_CELL_BITS};

/// LessThan256
pub type Rv32LessThan256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    LessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(TraceStep, TraceFiller)]
pub struct Rv32LessThan256Step(BaseStep);
pub type Rv32LessThan256Chip<F> =
    NewVmChipWrapper<F, Rv32LessThan256Air, Rv32LessThan256Step, MatrixRecordArena<F>>;

type BaseStep = LessThanStep<AdapterStep, INT256_NUM_LIMBS, RV32_CELL_BITS>;
type AdapterStep = Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

impl Rv32LessThan256Step {
    pub fn new(
        adapter: AdapterStep,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        offset: usize,
    ) -> Self {
        Self(BaseStep::new(adapter, bitwise_lookup_chip, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LessThanPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> StepExecutorE1<F> for Rv32LessThan256Step {
    fn pre_compute_size(&self) -> usize {
        size_of::<LessThanPreCompute>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> openvm_circuit::arch::Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1ExecutionCtx,
    {
        let data: &mut LessThanPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        let fn_ptr = match local_opcode {
            LessThanOpcode::SLT => execute_e1_impl::<_, _, false>,
            LessThanOpcode::SLTU => execute_e1_impl::<_, _, true>,
        };
        Ok(fn_ptr)
    }
}

impl<F: PrimeField32> StepExecutorE2<F> for Rv32LessThan256Step {
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LessThanPreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> openvm_circuit::arch::Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E2ExecutionCtx,
    {
        let data: &mut E2PreCompute<LessThanPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        let fn_ptr = match local_opcode {
            LessThanOpcode::SLT => execute_e2_impl::<_, _, false>,
            LessThanOpcode::SLTU => execute_e2_impl::<_, _, true>,
        };
        Ok(fn_ptr)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_U256: bool>(
    pre_compute: &LessThanPreCompute,
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let rs1_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c as u32);
    let rd_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs1 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let cmp_result = if IS_U256 {
        common::u256_lt(rs1, rs2)
    } else {
        common::i256_lt(rs1, rs2)
    };
    let mut rd = [0u8; INT256_NUM_LIMBS];
    rd[0] = cmp_result as u8;
    vm_state.vm_write(RV32_MEMORY_AS, u32::from_le_bytes(rd_ptr), &rd);

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_U256: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &LessThanPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_U256>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx, const IS_U256: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &E2PreCompute<LessThanPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_U256>(&pre_compute.data, vm_state);
}

impl Rv32LessThan256Step {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LessThanPreCompute,
    ) -> openvm_circuit::arch::Result<LessThanOpcode> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(InvalidInstruction(pc));
        }
        *data = LessThanPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        let local_opcode = LessThanOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LessThan256Opcode::CLASS_OFFSET),
        );
        Ok(local_opcode)
    }
}
