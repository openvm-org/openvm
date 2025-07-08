use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32BranchEqual256Opcode;
use openvm_circuit::arch::{
    execution_mode::E1ExecutionCtx, ExecuteFunc, ExecutionError::InvalidInstruction,
    MatrixRecordArena, NewVmChipWrapper, StepExecutorE1, VmAirWrapper, VmSegmentState,
};
use openvm_circuit_derive::{TraceFiller, TraceStep};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::{Rv32HeapBranchAdapterAir, Rv32HeapBranchAdapterStep};
use openvm_rv32im_circuit::{BranchEqualCoreAir, BranchEqualStep};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::INT256_NUM_LIMBS;

/// BranchEqual256
pub type Rv32BranchEqual256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchEqualCoreAir<INT256_NUM_LIMBS>,
>;
#[derive(TraceStep, TraceFiller)]
pub struct Rv32BranchEqual256Step(BaseStep);
pub type Rv32BranchEqual256Chip<F> =
    NewVmChipWrapper<F, Rv32BranchEqual256Air, Rv32BranchEqual256Step, MatrixRecordArena<F>>;

type BaseStep = BranchEqualStep<Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>, INT256_NUM_LIMBS>;
type AdapterStep = Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>;

impl Rv32BranchEqual256Step {
    pub fn new(adapter_step: AdapterStep, offset: usize, pc_step: u32) -> Self {
        Self(BaseStep::new(adapter_step, offset, pc_step))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BranchEqPreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

impl<F: PrimeField32> StepExecutorE1<F> for Rv32BranchEqual256Step {
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchEqPreCompute>()
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
        let data: &mut BranchEqPreCompute = data.borrow_mut();
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(InvalidInstruction(pc));
        }
        *data = BranchEqPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        let local_opcode = BranchEqualOpcode::from_usize(
            opcode.local_opcode_idx(Rv32BranchEqual256Opcode::CLASS_OFFSET),
        );
        let fn_ptr = match local_opcode {
            BranchEqualOpcode::BEQ => execute_e1_impl::<_, _, false>,
            BranchEqualOpcode::BNE => execute_e1_impl::<_, _, true>,
        };
        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_NE: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &BranchEqPreCompute = pre_compute.borrow();

    let rs1_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let cmp_result = u256_eq(rs1, rs2);
    if cmp_result ^ IS_NE {
        vm_state.pc = (vm_state.pc as isize + pre_compute.imm) as u32;
    } else {
        vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    }

    vm_state.instret += 1;
}

fn u256_eq(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    let rs1_u64: [u64; 4] = unsafe { std::mem::transmute(rs1) };
    let rs2_u64: [u64; 4] = unsafe { std::mem::transmute(rs2) };
    for i in 0..4 {
        if rs1_u64[i] != rs2_u64[i] {
            return false;
        }
    }
    true
}
