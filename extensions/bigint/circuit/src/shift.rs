use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32Shift256Opcode;
use openvm_circuit::arch::{
    execution_mode::E1ExecutionCtx, ExecuteFunc, ExecutionError::InvalidInstruction,
    MatrixRecordArena, NewVmChipWrapper, StepExecutorE1, VmAirWrapper, VmSegmentState,
};
use openvm_circuit_derive::{TraceFiller, TraceStep};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, var_range::SharedVariableRangeCheckerChip,
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32_adapters::{Rv32HeapAdapterAir, Rv32HeapAdapterStep};
use openvm_rv32im_circuit::{ShiftCoreAir, ShiftStep};
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{INT256_NUM_LIMBS, RV32_CELL_BITS};

/// Shift256
pub type Rv32Shift256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    ShiftCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
#[derive(TraceStep, TraceFiller)]
pub struct Rv32Shift256Step(BaseStep);
pub type Rv32Shift256Chip<F> =
    NewVmChipWrapper<F, Rv32Shift256Air, Rv32Shift256Step, MatrixRecordArena<F>>;

type BaseStep = ShiftStep<AdapterStep, INT256_NUM_LIMBS, RV32_CELL_BITS>;
type AdapterStep = Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>;

impl Rv32Shift256Step {
    pub fn new(
        adapter: AdapterStep,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
        offset: usize,
    ) -> Self {
        Self(BaseStep::new(
            adapter,
            bitwise_lookup_chip,
            range_checker_chip,
            offset,
        ))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> StepExecutorE1<F> for Rv32Shift256Step {
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftPreCompute>()
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
        let data: &mut ShiftPreCompute = data.borrow_mut();
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
        *data = ShiftPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        let local_opcode =
            ShiftOpcode::from_usize(opcode.local_opcode_idx(Rv32Shift256Opcode::CLASS_OFFSET));
        let fn_ptr = match local_opcode {
            ShiftOpcode::SLL => execute_e1_impl::<_, _, SllOp>,
            ShiftOpcode::SRA => execute_e1_impl::<_, _, SraOp>,
            ShiftOpcode::SRL => execute_e1_impl::<_, _, SrlOp>,
        };
        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, OP: ShiftOp>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &ShiftPreCompute = pre_compute.borrow();
    let rs1_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c as u32);
    let rd_ptr = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs1 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 = vm_state.vm_read::<u8, INT256_NUM_LIMBS>(RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let rd = OP::compute(rs1, rs2);
    vm_state.vm_write(RV32_MEMORY_AS, u32::from_le_bytes(rd_ptr), &rd);
    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

trait ShiftOp {
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS];
}
struct SllOp;
struct SrlOp;
struct SraOp;
impl ShiftOp for SllOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = unsafe { std::mem::transmute(rs1) };
        let rs2_u64: [u64; 4] = unsafe { std::mem::transmute(rs2) };
        let mut rd = [0u64; 4];
        if !(rs2_u64[0] > 0xff || rs2_u64[1] > 0 || rs2_u64[2] > 0 || rs2_u64[3] > 0) {
            let shift = rs2_u64[0] as u32;
            let index_offset = (shift / u64::BITS) as usize;
            let bit_offset = shift % u64::BITS;
            let mut carry = 0u64;
            for i in index_offset..4 {
                let curr = rs1_u64[i - index_offset];
                rd[i] = (curr << bit_offset) + carry;
                carry = curr >> (u64::BITS - bit_offset);
            }
        }
        unsafe { std::mem::transmute(rd) }
    }
}
impl ShiftOp for SrlOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        shift_right(rs1, rs2, 0)
    }
}
impl ShiftOp for SraOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        if rs1[INT256_NUM_LIMBS - 1] & 0x80 > 0 {
            shift_right(rs1, rs2, u64::MAX)
        } else {
            shift_right(rs1, rs2, 0)
        }
    }
}

#[inline(always)]
fn shift_right(
    rs1: [u8; INT256_NUM_LIMBS],
    rs2: [u8; INT256_NUM_LIMBS],
    init_value: u64,
) -> [u8; INT256_NUM_LIMBS] {
    let rs1_u64: [u64; 4] = unsafe { std::mem::transmute(rs1) };
    let rs2_u64: [u64; 4] = unsafe { std::mem::transmute(rs2) };
    let mut rd = [init_value; 4];
    if !(rs2_u64[0] > 0xff || rs2_u64[1] > 0 || rs2_u64[2] > 0 || rs2_u64[3] > 0) {
        let shift = rs2_u64[0] as u32;
        let index_offset = (shift / u64::BITS) as usize;
        let bit_offset = shift % u64::BITS;
        let mut carry = 0u64;
        for i in (index_offset..4).rev() {
            let curr = rs1_u64[i];
            rd[i - index_offset] = (curr >> bit_offset) + carry;
            carry = curr << (u64::BITS - bit_offset);
        }
    }
    unsafe { std::mem::transmute(rd) }
}
