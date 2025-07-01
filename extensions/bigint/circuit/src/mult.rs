use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32Mul256Opcode;
use openvm_circuit::arch::{
    execution_mode::E1E2ExecutionCtx, ExecuteFunc, ExecutionError::InvalidInstruction,
    MatrixRecordArena, NewVmChipWrapper, StepExecutorE1, VmAirWrapper, VmSegmentState,
};
use openvm_circuit_derive::{TraceFiller, TraceStep};
use openvm_circuit_primitives::range_tuple::SharedRangeTupleCheckerChip;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::{Rv32HeapAdapterAir, Rv32HeapAdapterStep};
use openvm_rv32im_circuit::{MultiplicationCoreAir, MultiplicationStep};
use openvm_rv32im_transpiler::MulOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{INT256_NUM_LIMBS, RV32_CELL_BITS};

/// Multiplication256
pub type Rv32Multiplication256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    MultiplicationCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(TraceStep, TraceFiller)]
pub struct Rv32Multiplication256Step(BaseStep);
pub type Rv32Multiplication256Chip<F> =
    NewVmChipWrapper<F, Rv32Multiplication256Air, Rv32Multiplication256Step, MatrixRecordArena<F>>;

type BaseStep = MultiplicationStep<AdapterStep, INT256_NUM_LIMBS, RV32_CELL_BITS>;
type AdapterStep = Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

impl Rv32Multiplication256Step {
    pub fn new(
        adapter: AdapterStep,
        range_tuple_chip: SharedRangeTupleCheckerChip<2>,
        offset: usize,
    ) -> Self {
        Self(BaseStep::new(adapter, range_tuple_chip, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MultPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> StepExecutorE1<F> for Rv32Multiplication256Step {
    fn pre_compute_size(&self) -> usize {
        size_of::<MultPreCompute>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> openvm_circuit::arch::Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let data: &mut MultPreCompute = data.borrow_mut();
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
        let local_opcode =
            MulOpcode::from_usize(opcode.local_opcode_idx(Rv32Mul256Opcode::CLASS_OFFSET));
        assert_eq!(local_opcode, MulOpcode::MUL);
        *data = MultPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        Ok(execute_e1_impl)
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &MultPreCompute = pre_compute.borrow();
    let rs1_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c as u32);
    let rd_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs1 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let rd = u256_mul(rs1, rs2);
    vm_state.vm_write(RV32_REGISTER_AS, u32::from_le_bytes(rd_ptr), &rd);

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}

#[inline(always)]
fn u256_mul(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
    let rs1_u64: [u32; 8] = unsafe { std::mem::transmute(rs1) };
    let rs2_u64: [u32; 8] = unsafe { std::mem::transmute(rs2) };
    let mut rd = [0u32; 8];
    for i in 0..8 {
        let mut carry = 0u64;
        for j in 0..(8 - i) {
            let res = rs1_u64[i] as u64 * rs2_u64[j] as u64 + rd[i + j] as u64 + carry;
            rd[i + j] = res as u32;
            carry = res >> 32;
        }
    }
    unsafe { std::mem::transmute(rd) }
}
