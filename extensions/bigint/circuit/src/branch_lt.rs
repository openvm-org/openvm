use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32BranchEqual256Opcode;
use openvm_circuit::arch::{
    execution_mode::E1ExecutionCtx, ExecuteFunc, ExecutionError::InvalidInstruction,
    MatrixRecordArena, NewVmChipWrapper, StepExecutorE1, VmAirWrapper, VmSegmentState,
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
use openvm_rv32_adapters::{Rv32HeapBranchAdapterAir, Rv32HeapBranchAdapterStep};
use openvm_rv32im_circuit::{BranchLessThanCoreAir, BranchLessThanStep};
use openvm_rv32im_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    common::{i256_lt, u256_lt},
    INT256_NUM_LIMBS, RV32_CELL_BITS,
};

/// BranchLessThan256
pub type Rv32BranchLessThan256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchLessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(TraceStep, TraceFiller)]
pub struct Rv32BranchLessThan256Step(BaseStep);
pub type Rv32BranchLessThan256Chip<F> =
    NewVmChipWrapper<F, Rv32BranchLessThan256Air, Rv32BranchLessThan256Step, MatrixRecordArena<F>>;

type BaseStep = BranchLessThanStep<
    Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>,
    INT256_NUM_LIMBS,
    RV32_CELL_BITS,
>;
type AdapterStep = Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>;

impl Rv32BranchLessThan256Step {
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
struct BranchLtPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<F: PrimeField32> StepExecutorE1<F> for Rv32BranchLessThan256Step {
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchLtPreCompute>()
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
        let data: &mut BranchLtPreCompute = data.borrow_mut();
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
        *data = BranchLtPreCompute {
            c: c.as_canonical_u32(),
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        let local_opcode = BranchLessThanOpcode::from_usize(
            opcode.local_opcode_idx(Rv32BranchEqual256Opcode::CLASS_OFFSET),
        );
        let fn_ptr = match local_opcode {
            BranchLessThanOpcode::BLT => execute_e1_impl::<_, _, BltOp>,
            BranchLessThanOpcode::BLTU => execute_e1_impl::<_, _, BltuOp>,
            BranchLessThanOpcode::BGE => execute_e1_impl::<_, _, BgeOp>,
            BranchLessThanOpcode::BGEU => execute_e1_impl::<_, _, BgeuOp>,
        };
        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, OP: BranchLessThanOp>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &BranchLtPreCompute = pre_compute.borrow();
    let rs1_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let cmp_result = OP::compute(rs1, rs2);
    if cmp_result {
        vm_state.pc += pre_compute.c;
    } else {
        vm_state.pc += DEFAULT_PC_STEP;
    }
    vm_state.instret += 1;
}

trait BranchLessThanOp {
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool;
}
struct BltOp;
struct BltuOp;
struct BgeOp;
struct BgeuOp;

impl BranchLessThanOp for BltOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        i256_lt(rs1, rs2)
    }
}
impl BranchLessThanOp for BltuOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        u256_lt(rs1, rs2)
    }
}
impl BranchLessThanOp for BgeOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        !i256_lt(rs1, rs2)
    }
}
impl BranchLessThanOp for BgeuOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
        !u256_lt(rs1, rs2)
    }
}
