use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::Rv32LessThan256Opcode;
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
        let fn_ptr = match local_opcode {
            LessThanOpcode::SLT => execute_e1_impl::<_, _, false>,
            LessThanOpcode::SLTU => execute_e1_impl::<_, _, true>,
        };
        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_U256: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &LessThanPreCompute = pre_compute.borrow();

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
    vm_state.vm_write(RV32_REGISTER_AS, u32::from_le_bytes(rd_ptr), &rd);

    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}
