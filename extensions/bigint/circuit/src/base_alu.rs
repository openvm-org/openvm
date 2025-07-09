use std::{
    borrow::{Borrow, BorrowMut},
    mem::transmute,
};

use openvm_bigint_transpiler::Rv32BaseAlu256Opcode;
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
use openvm_rv32_adapters::{Rv32HeapAdapterAir, Rv32HeapAdapterStep};
use openvm_rv32im_circuit::{BaseAluCoreAir, BaseAluStep};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{INT256_NUM_LIMBS, RV32_CELL_BITS};

pub type Rv32BaseAlu256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    BaseAluCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

#[derive(TraceStep, TraceFiller)]
pub struct Rv32BaseAlu256Step(BaseStep);
pub type Rv32BaseAlu256Chip<F> =
    NewVmChipWrapper<F, Rv32BaseAlu256Air, Rv32BaseAlu256Step, MatrixRecordArena<F>>;

type BaseStep = BaseAluStep<AdapterStep, INT256_NUM_LIMBS, RV32_CELL_BITS>;
type AdapterStep = Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

impl Rv32BaseAlu256Step {
    pub fn new(
        adapter: AdapterStep,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        offset: usize,
    ) -> Self {
        Self(BaseAluStep::new(adapter, bitwise_lookup_chip, offset))
    }
}

#[derive(AlignedBytesBorrow)]
struct BaseAluPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> StepExecutorE1<F> for Rv32BaseAlu256Step {
    fn pre_compute_size(&self) -> usize {
        size_of::<BaseAluPreCompute>()
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
        let data: &mut BaseAluPreCompute = data.borrow_mut();
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
        *data = BaseAluPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        let local_opcode =
            BaseAluOpcode::from_usize(opcode.local_opcode_idx(Rv32BaseAlu256Opcode::CLASS_OFFSET));
        let fn_ptr = match local_opcode {
            BaseAluOpcode::ADD => execute_e1_impl::<_, _, AddOp>,
            BaseAluOpcode::SUB => execute_e1_impl::<_, _, SubOp>,
            BaseAluOpcode::XOR => execute_e1_impl::<_, _, XorOp>,
            BaseAluOpcode::OR => execute_e1_impl::<_, _, OrOp>,
            BaseAluOpcode::AND => execute_e1_impl::<_, _, AndOp>,
        };
        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, OP: AluOp>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &BaseAluPreCompute = pre_compute.borrow();
    let rs1_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c as u32);
    let rd_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs1 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let rd = <OP as AluOp>::compute(rs1, rs2);
    vm_state.vm_write(RV32_MEMORY_AS, u32::from_le_bytes(rd_ptr), &rd);
    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

trait AluOp {
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS];
}
struct AddOp;
struct SubOp;
struct XorOp;
struct OrOp;
struct AndOp;
impl AluOp for AddOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = unsafe { transmute(rs1) };
        let rs2_u64: [u64; 4] = unsafe { transmute(rs2) };
        let mut rd_u64 = [0u64; 4];
        let (res, mut overflow) = rs1_u64[0].overflowing_add(rs2_u64[0]);
        rd_u64[0] = res;
        // Compiler will expand this loop.
        for i in 1..4 {
            let (res1, c1) = rs1_u64[i].overflowing_add(rs2_u64[i]);
            let (res2, c2) = res1.overflowing_add(overflow as u64);
            overflow = c1 || c2;
            rd_u64[i] = res2;
        }
        unsafe { transmute(rd_u64) }
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = unsafe { transmute(rs1) };
        let rs2_u64: [u64; 4] = unsafe { transmute(rs2) };
        let mut rd_u64 = [0u64; 4];
        let (res, mut overflow) = rs1_u64[0].overflowing_sub(rs2_u64[0]);
        rd_u64[0] = res;
        // Compiler will expand this loop.
        for i in 1..4 {
            let (res1, c1) = rs1_u64[i].overflowing_sub(rs2_u64[i]);
            let (res2, c2) = res1.overflowing_sub(overflow as u64);
            overflow = c1 || c2;
            rd_u64[i] = res2;
        }
        unsafe { transmute(rd_u64) }
    }
}
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = unsafe { transmute(rs1) };
        let rs2_u64: [u64; 4] = unsafe { transmute(rs2) };
        let mut rd_u64 = [0u64; 4];
        // Compiler will expand this loop.
        for i in 0..4 {
            rd_u64[i] = rs1_u64[i] ^ rs2_u64[i];
        }
        unsafe { transmute(rd_u64) }
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = unsafe { transmute(rs1) };
        let rs2_u64: [u64; 4] = unsafe { transmute(rs2) };
        let mut rd_u64 = [0u64; 4];
        // Compiler will expand this loop.
        for i in 0..4 {
            rd_u64[i] = rs1_u64[i] | rs2_u64[i];
        }
        unsafe { transmute(rd_u64) }
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = unsafe { transmute(rs1) };
        let rs2_u64: [u64; 4] = unsafe { transmute(rs2) };
        let mut rd_u64 = [0u64; 4];
        // Compiler will expand this loop.
        for i in 0..4 {
            rd_u64[i] = rs1_u64[i] & rs2_u64[i];
        }
        unsafe { transmute(rd_u64) }
    }
}
