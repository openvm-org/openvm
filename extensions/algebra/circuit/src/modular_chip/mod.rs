use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use derive_more::{Deref, DerefMut};
use num_bigint::BigUint;
use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::{
    arch::{
        execution::ExecuteFunc,
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        DynArray, E2PreCompute,
        ExecutionError::{self, InvalidInstruction},
        InsExecutorE1, InsExecutorE2, Result, VmAirWrapper, VmChipWrapper, VmSegmentState,
    },
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpr, FieldExpressionCoreAir, FieldExpressionFiller,
    FieldExpressionStep,
};
use openvm_rv32_adapters::{
    Rv32IsEqualModAdapterAir, Rv32IsEqualModAdapterFiller, Rv32IsEqualModAdapterStep,
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller, Rv32VecHeapAdapterStep,
};
use openvm_stark_backend::p3_field::PrimeField32;

use self::fields::{field_operation, get_field_type_from_modulus, FieldType, Operation};

pub mod fields;
mod is_eq;
pub use is_eq::*;
mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

#[cfg(test)]
mod tests;

pub type ModularAir<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

#[derive(Clone, InstructionExecutor, Deref, DerefMut)]
pub struct ModularStep<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    FieldExpressionStep<Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
);

pub type ModularChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = VmChipWrapper<
    F,
    FieldExpressionFiller<Rv32VecHeapAdapterFiller<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
>;

// Must have TOTAL_LIMBS = NUM_LANES * LANE_SIZE
pub type ModularIsEqualAir<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmAirWrapper<
    Rv32IsEqualModAdapterAir<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    ModularIsEqualCoreAir<TOTAL_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

#[derive(Clone, InstructionExecutor)]
pub struct VmModularIsEqualStep<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
>(
    ModularIsEqualStep<
        Rv32IsEqualModAdapterStep<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        TOTAL_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);

pub type ModularIsEqualChip<
    F,
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmChipWrapper<
    F,
    ModularIsEqualFiller<
        Rv32IsEqualModAdapterFiller<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        TOTAL_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ModularPreCompute<'a> {
    expr: &'a FieldExpr,
    rs_addrs: [u8; 2],
    a: u8,
    flag_idx: u8,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> ModularStep<BLOCKS, BLOCK_SIZE> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ModularPreCompute<'a>,
    ) -> Result<(bool, Operation)> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        // Validate instruction format
        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        if d != RV32_REGISTER_AS || e != RV32_MEMORY_AS {
            return Err(InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.0.offset);

        // Pre-compute flag_idx
        let needs_setup = self.0.expr.needs_setup();
        let mut flag_idx = self.0.expr.num_flags() as u8;
        if needs_setup {
            // Find which opcode this is in our local_opcode_idx list
            if let Some(opcode_position) = self
                .0
                .local_opcode_idx
                .iter()
                .position(|&idx| idx == local_opcode)
            {
                // If this is NOT the last opcode (setup), get the corresponding flag_idx
                if opcode_position < self.0.opcode_flag_idx.len() {
                    flag_idx = self.0.opcode_flag_idx[opcode_position] as u8;
                }
            }
        }

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = ModularPreCompute {
            expr: &self.0.expr,
            rs_addrs,
            a: a as u8,
            flag_idx,
        };

        let is_setup = local_opcode == Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize
            || local_opcode == Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize;

        let op = match local_opcode {
            x if x == Rv32ModularArithmeticOpcode::ADD as usize => Operation::Add,
            x if x == Rv32ModularArithmeticOpcode::SUB as usize => Operation::Sub,
            x if x == Rv32ModularArithmeticOpcode::MUL as usize => Operation::Mul,
            x if x == Rv32ModularArithmeticOpcode::DIV as usize => Operation::Div,
            _ => unreachable!(),
        };

        Ok((is_setup, op))
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> InsExecutorE1<F>
    for ModularStep<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<ModularPreCompute>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1ExecutionCtx,
    {
        let pre_compute: &mut ModularPreCompute = data.borrow_mut();

        let (is_setup, op) = self.pre_compute_impl(pc, inst, pre_compute)?;

        if is_setup {
            Ok(execute_e1_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(field_type) = {
            let modulus = &pre_compute.expr.builder.prime;
            get_field_type_from_modulus(modulus)
        } {
            match (field_type, op) {
                (FieldType::BN254, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BN254, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BN254, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BN254, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::BLS12_381, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BLS12_381, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BLS12_381, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BLS12_381, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::K256, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::K256, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::K256, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::K256, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::P256, Operation::Add) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::P256, Operation::Sub) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::P256, Operation::Mul) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::P256, Operation::Div) => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Div as u8 },
                >),
            }
        } else {
            match op {
                Operation::Add => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Add as u8 },
                >),
                Operation::Sub => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Sub as u8 },
                >),
                Operation::Mul => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Mul as u8 },
                >),
                Operation::Div => Ok(execute_e1_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Div as u8 },
                >),
            }
        }
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> InsExecutorE2<F>
    for ModularStep<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<ModularPreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E2ExecutionCtx,
    {
        let pre_compute: &mut E2PreCompute<ModularPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let (is_setup, op) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        if is_setup {
            Ok(execute_e2_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(field_type) = {
            let modulus = &pre_compute.data.expr.builder.prime;
            get_field_type_from_modulus(modulus)
        } {
            match (field_type, op) {
                (FieldType::BN254, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BN254, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BN254, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BN254, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BN254 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::BLS12_381, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::BLS12_381, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::BLS12_381, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::BLS12_381, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::BLS12_381 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::K256, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::K256, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::K256, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::K256, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::K256 as u8 },
                    { Operation::Div as u8 },
                >),
                (FieldType::P256, Operation::Add) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Add as u8 },
                >),
                (FieldType::P256, Operation::Sub) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Sub as u8 },
                >),
                (FieldType::P256, Operation::Mul) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Mul as u8 },
                >),
                (FieldType::P256, Operation::Div) => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { FieldType::P256 as u8 },
                    { Operation::Div as u8 },
                >),
            }
        } else {
            match op {
                Operation::Add => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Add as u8 },
                >),
                Operation::Sub => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Sub as u8 },
                >),
                Operation::Mul => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Mul as u8 },
                >),
                Operation::Div => Ok(execute_e2_impl::<
                    _,
                    _,
                    BLOCKS,
                    BLOCK_SIZE,
                    { u8::MAX },
                    { Operation::Div as u8 },
                >),
            }
        }
    }
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ModularPreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, FIELD_TYPE, OP>(pre_compute, vm_state);
}

unsafe fn execute_e1_setup_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ModularPreCompute = pre_compute.borrow();
    execute_e12_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ModularPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, FIELD_TYPE, OP>(&pre_compute.data, vm_state);
}

unsafe fn execute_e2_setup_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ModularPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>(&pre_compute.data, vm_state);
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const FIELD_TYPE: u8,
    const OP: u8,
>(
    pre_compute: &ModularPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read memory values for both inputs
    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });

    let output_data = if FIELD_TYPE == u8::MAX {
        let read_data: DynArray<u8> = read_data.into();
        run_field_expression_precomputed::<true>(
            pre_compute.expr,
            pre_compute.flag_idx as usize,
            &read_data.0,
        )
        .into()
    } else {
        field_operation::<FIELD_TYPE, BLOCKS, BLOCK_SIZE, OP>(read_data)
    };

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    // Write output data to memory
    for (i, block) in output_data.into_iter().enumerate() {
        vm_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

unsafe fn execute_e12_setup_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    pre_compute: &ModularPreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    // Read the first input (which should be the prime)
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read the first input's data as the setup input
    let setup_input_data: [[u8; BLOCK_SIZE]; BLOCKS] = {
        let address = rs_vals[0];
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    };

    // Extract field element as the prime
    let input_prime = BigUint::from_bytes_le(setup_input_data.as_flattened());

    if input_prime != pre_compute.expr.prime {
        vm_state.exit_code = Err(ExecutionError::Fail { pc: vm_state.pc });
        return;
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}
