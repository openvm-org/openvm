use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use derive_more::derive::{Deref, DerefMut};
use num_bigint::BigUint;
use openvm_algebra_circuit::FieldExprVecHeapStep;
use openvm_circuit::{
    arch::{
        execution::ExecuteFunc,
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        E2PreCompute, ExecutionBridge,
        ExecutionError::{self, InvalidInstruction},
        InsExecutorE1, InsExecutorE2, Result, VmSegmentState,
    },
    system::memory::{
        offline_checker::MemoryBridge, online::GuestMemory, SharedMemoryHelper, POINTER_MAX_BITS,
    },
};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_CELL_BITS,
};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldExpressionFiller,
};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller, Rv32VecHeapAdapterStep,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{curves::get_curve_type_from_modulus, WeierstrassAir, WeierstrassChip};
use crate::weierstrass_chip::curves::{ec_add_ne, CurveType};

// Assumes that (x1, y1), (x2, y2) both lie on the curve and are not the identity point.
// Further assumes that x1, x2 are not equal in the coordinate field.
pub fn ec_add_ne_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
    let mut x3 = lambda.square() - x1.clone() - x2;
    x3.save_output();
    let mut y3 = lambda * (x1 - x3.clone()) - y1;
    y3.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new(builder, range_bus, true)
}

/// BLOCK_SIZE: how many cells do we read at a time, must be a power of 2.
/// BLOCKS: how many blocks do we need to represent one input or output
/// For example, for bls12_381, BLOCK_SIZE = 16, each element has 3 blocks and with two elements per
/// input AffinePoint, BLOCKS = 6. For secp256k1, BLOCK_SIZE = 32, BLOCKS = 2.
#[derive(Clone, InstructionExecutor, Deref, DerefMut)]
pub struct EcAddNeStep<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub FieldExprVecHeapStep<2, BLOCKS, BLOCK_SIZE>,
);

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
) -> (FieldExpr, Vec<usize>) {
    let expr = ec_add_ne_expr(config, range_checker_bus);

    let local_opcode_idx = vec![
        Rv32WeierstrassOpcode::EC_ADD_NE as usize,
        Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
    ];

    (expr, local_opcode_idx)
}

pub fn get_ec_addne_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
) -> WeierstrassAir<2, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus);
    WeierstrassAir::new(
        Rv32VecHeapAdapterAir::new(
            exec_bridge,
            mem_bridge,
            bitwise_lookup_bus,
            pointer_max_bits,
        ),
        FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
    )
}

pub fn get_ec_addne_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
) -> EcAddNeStep<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus);
    EcAddNeStep(FieldExprVecHeapStep::new(
        Rv32VecHeapAdapterStep::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        vec![],
        "EcAddNe",
    ))
}

pub fn get_ec_addne_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
) -> WeierstrassChip<F, 2, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus());
    WeierstrassChip::new(
        FieldExpressionFiller::new(
            Rv32VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            false,
        ),
        mem_helper,
    )
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct EcAddNePreCompute<'a> {
    modulus: &'a BigUint,
    rs_addrs: [u8; 2],
    a: u8,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> EcAddNeStep<BLOCKS, BLOCK_SIZE> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EcAddNePreCompute<'a>,
    ) -> Result<bool> {
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

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = EcAddNePreCompute {
            a: a as u8,
            rs_addrs,
            modulus: &self.expr.builder.prime,
        };

        let local_opcode = opcode.local_opcode_idx(self.offset);
        let is_setup = local_opcode == Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize;

        Ok(is_setup)
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> InsExecutorE1<F>
    for EcAddNeStep<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EcAddNePreCompute>()
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
        let pre_compute: &mut EcAddNePreCompute = data.borrow_mut();

        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        if is_setup {
            Ok(execute_e1_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(curve_type) = get_curve_type_from_modulus(pre_compute.modulus) {
            match curve_type {
                CurveType::K256 => {
                    Ok(execute_e1_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::K256 as u8 }>)
                }
                CurveType::P256 => {
                    Ok(execute_e1_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::P256 as u8 }>)
                }
                CurveType::BN254 => {
                    Ok(execute_e1_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::BN254 as u8 }>)
                }
                CurveType::BLS12_381 => {
                    Ok(execute_e1_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::BLS12_381 as u8 }>)
                }
            }
        } else {
            Ok(execute_e1_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }>)
        }
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> InsExecutorE2<F>
    for EcAddNeStep<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<EcAddNePreCompute>>()
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
        let pre_compute: &mut E2PreCompute<EcAddNePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let is_setup = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        if is_setup {
            Ok(execute_e2_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(curve_type) = get_curve_type_from_modulus(pre_compute.data.modulus) {
            match curve_type {
                CurveType::K256 => {
                    Ok(execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::K256 as u8 }>)
                }
                CurveType::P256 => {
                    Ok(execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::P256 as u8 }>)
                }
                CurveType::BN254 => {
                    Ok(execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::BN254 as u8 }>)
                }
                CurveType::BLS12_381 => {
                    Ok(execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::BLS12_381 as u8 }>)
                }
            }
        } else {
            Ok(execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { u8::MAX }>)
        }
    }
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &EcAddNePreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE>(pre_compute, vm_state);
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
    let pre_compute: &EcAddNePreCompute = pre_compute.borrow();

    execute_e12_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<EcAddNePreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, BLOCKS, BLOCK_SIZE, CURVE_TYPE>(&pre_compute.data, vm_state);
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
    let pre_compute: &E2PreCompute<EcAddNePreCompute> = pre_compute.borrow();
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
    const CURVE_TYPE: u8,
>(
    pre_compute: &EcAddNePreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read memory values for both points
    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });

    let output_data = match CURVE_TYPE {
        x if x == CurveType::K256 as u8 => ec_add_ne::<0, BLOCKS, BLOCK_SIZE>(read_data),
        x if x == CurveType::P256 as u8 => ec_add_ne::<1, BLOCKS, BLOCK_SIZE>(read_data),
        x if x == CurveType::BN254 as u8 => ec_add_ne::<2, BLOCKS, BLOCK_SIZE>(read_data),
        x if x == CurveType::BLS12_381 as u8 => ec_add_ne::<3, BLOCKS, BLOCK_SIZE>(read_data),

        _ => ec_add_ne_generic::<BLOCKS, BLOCK_SIZE>(read_data, pre_compute.modulus),
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
    pre_compute: &EcAddNePreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    // Read the first input (which should be the prime)
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read the first point's data as the setup input
    let setup_input_data: [[u8; BLOCK_SIZE]; BLOCKS] = {
        let address = rs_vals[0];
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    };

    // Extract first field element as the prime
    let prime_bytes: Vec<u8> = setup_input_data[..BLOCKS / 2]
        .iter()
        .flatten()
        .copied()
        .collect();
    let input_prime = BigUint::from_bytes_le(&prime_bytes);

    if input_prime != *pre_compute.modulus {
        vm_state.exit_code = Err(ExecutionError::Fail { pc: vm_state.pc });
        return;
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

// Assumes that (x1, y1), (x2, y2) both lie on the curve and are not the identity point.
// Further assumes that x1, x2 are not equal in the coordinate field.
fn ec_add_ne_generic<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
    field_modulus: &BigUint,
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let field_element_bytes = BLOCKS * BLOCK_SIZE;
    let half_bytes = field_element_bytes / 2;

    // Extract coordinates from input data
    let x1_bytes: Vec<u8> = input_data[0][..BLOCKS / 2]
        .iter()
        .flatten()
        .copied()
        .collect();
    let y1_bytes: Vec<u8> = input_data[0][BLOCKS / 2..]
        .iter()
        .flatten()
        .copied()
        .collect();
    let x2_bytes: Vec<u8> = input_data[1][..BLOCKS / 2]
        .iter()
        .flatten()
        .copied()
        .collect();
    let y2_bytes: Vec<u8> = input_data[1][BLOCKS / 2..]
        .iter()
        .flatten()
        .copied()
        .collect();

    // Convert to BigUint for modular arithmetic
    let x1 = BigUint::from_bytes_le(&x1_bytes);
    let y1 = BigUint::from_bytes_le(&y1_bytes);
    let x2 = BigUint::from_bytes_le(&x2_bytes);
    let y2 = BigUint::from_bytes_le(&y2_bytes);

    // Elliptic curve point addition formula:
    // lambda = (y2 - y1) / (x2 - x1) mod p
    // x3 = lambda^2 - x1 - x2 mod p
    // y3 = lambda * (x1 - x3) - y1 mod p

    // Calculate lambda = (y2 - y1) / (x2 - x1) mod p
    let y_diff = if y2 >= y1 {
        (&y2 - &y1) % field_modulus
    } else {
        (field_modulus + &y2 - &y1) % field_modulus
    };

    let x_diff = if x2 >= x1 {
        (&x2 - &x1) % field_modulus
    } else {
        (field_modulus + &x2 - &x1) % field_modulus
    };

    // Calculate modular inverse of x_diff using Extended Euclidean Algorithm
    let x_diff_inv = x_diff
        .modinv(field_modulus)
        .expect("Modular inverse should exist for valid EC points");
    let lambda = (&y_diff * &x_diff_inv) % field_modulus;

    // Calculate x3 = lambda^2 - x1 - x2 mod p
    let lambda_squared = (&lambda * &lambda) % field_modulus;
    let x3 = (&lambda_squared + field_modulus + field_modulus - &x1 - &x2) % field_modulus;

    // Calculate y3 = lambda * (x1 - x3) - y1 mod p
    let x1_minus_x3 = if x1 >= x3 {
        (&x1 - &x3) % field_modulus
    } else {
        (field_modulus + &x1 - &x3) % field_modulus
    };

    let y3 = {
        let temp = (&lambda * &x1_minus_x3) % field_modulus;
        if temp >= y1 {
            (&temp - &y1) % field_modulus
        } else {
            (field_modulus + &temp - &y1) % field_modulus
        }
    };

    // Convert results back to byte representation
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];

    // Store x3 in first half of blocks
    let x3_bytes = x3.to_bytes_le();
    let x3_len = x3_bytes.len().min(half_bytes);
    for (i, &byte) in x3_bytes[..x3_len].iter().enumerate() {
        let block_idx = i / BLOCK_SIZE;
        let byte_idx = i % BLOCK_SIZE;
        if block_idx < BLOCKS / 2 {
            output[block_idx][byte_idx] = byte;
        }
    }

    // Store y3 in second half of blocks
    let y3_bytes = y3.to_bytes_le();
    let y3_len = y3_bytes.len().min(half_bytes);
    for (i, &byte) in y3_bytes[..y3_len].iter().enumerate() {
        let block_idx = (BLOCKS / 2) + (i / BLOCK_SIZE);
        let byte_idx = i % BLOCK_SIZE;
        if block_idx < BLOCKS {
            output[block_idx][byte_idx] = byte;
        }
    }

    output
}
