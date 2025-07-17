use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use derive_more::derive::{Deref, DerefMut};
use num_bigint::BigUint;
use num_traits::One;
use openvm_algebra_circuit::FieldExprVecHeapStep;
use openvm_circuit::{
    arch::{
        execution::ExecuteFunc,
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        E2PreCompute, ExecutionBridge,
        ExecutionError::InvalidInstruction,
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
    FieldVariable,
};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller, Rv32VecHeapAdapterStep,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{curves::get_curve_type, WeierstrassAir, WeierstrassChip};
use crate::weierstrass_chip::curves::{ec_double, CurveType};

pub fn ec_double_ne_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let is_double_flag = (*builder).borrow_mut().new_flag();
    // We need to prevent divide by zero when not double flag
    // (equivalently, when it is the setup opcode)
    let lambda_denom = FieldVariable::select(
        is_double_flag,
        &y1.int_mul(2),
        &ExprBuilder::new_const(builder.clone(), BigUint::one()),
    );
    let mut lambda = (x1.square().int_mul(3) + a) / lambda_denom;
    let mut x3 = lambda.square() - x1.int_mul(2);
    x3.save_output();
    let mut y3 = lambda * (x1 - x3.clone()) - y1;
    y3.save_output();

    let builder = (*builder).borrow().clone();
    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_biguint])
}

/// BLOCK_SIZE: how many cells do we read at a time, must be a power of 2.
/// BLOCKS: how many blocks do we need to represent one input or output
/// For example, for bls12_381, BLOCK_SIZE = 16, each element has 3 blocks and with two elements per
/// input AffinePoint, BLOCKS = 6. For secp256k1, BLOCK_SIZE = 32, BLOCKS = 2.
#[derive(Clone, InstructionExecutor, Deref, DerefMut)]
pub struct EcDoubleStep<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub FieldExprVecHeapStep<1, BLOCKS, BLOCK_SIZE>,
);

fn gen_base_expr(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> (FieldExpr, Vec<usize>) {
    let expr = ec_double_ne_expr(config, range_checker_bus, a_biguint);

    let local_opcode_idx = vec![
        Rv32WeierstrassOpcode::EC_DOUBLE as usize,
        Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
    ];

    (expr, local_opcode_idx)
}

#[allow(clippy::too_many_arguments)]
pub fn get_ec_double_air<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    exec_bridge: ExecutionBridge,
    mem_bridge: MemoryBridge,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
) -> WeierstrassAir<1, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint);
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

pub fn get_ec_double_step<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
) -> EcDoubleStep<BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker_bus, a_biguint);
    EcDoubleStep(FieldExprVecHeapStep::new(
        Rv32VecHeapAdapterStep::new(pointer_max_bits),
        expr,
        offset,
        local_opcode_idx,
        vec![],
        "EcDouble",
    ))
}

pub fn get_ec_double_chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
    a_biguint: BigUint,
) -> WeierstrassChip<F, 1, BLOCKS, BLOCK_SIZE> {
    let (expr, local_opcode_idx) = gen_base_expr(config, range_checker.bus(), a_biguint);
    WeierstrassChip::new(
        FieldExpressionFiller::new(
            Rv32VecHeapAdapterFiller::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            local_opcode_idx,
            vec![],
            range_checker,
            true,
        ),
        mem_helper,
    )
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct EcDoublePreCompute<'a> {
    a: u8,
    rs_addrs: [u8; 1],
    modulus: &'a BigUint,
    a_coeff: &'a BigUint,
}

impl<'a, const BLOCKS: usize, const BLOCK_SIZE: usize> EcDoubleStep<BLOCKS, BLOCK_SIZE> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EcDoublePreCompute<'a>,
    ) -> Result<bool> {
        let Instruction {
            opcode, a, b, d, e, ..
        } = inst;

        // Validate instruction format
        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        if d != RV32_REGISTER_AS || e != RV32_MEMORY_AS {
            return Err(InvalidInstruction(pc));
        }

        let rs_addrs = [b as u8];
        *data = EcDoublePreCompute {
            a: a as u8,
            rs_addrs,
            modulus: &self.expr.builder.prime,
            a_coeff: &self.expr.setup_values[0],
        };

        let local_opcode = opcode.local_opcode_idx(self.offset);
        let is_setup = local_opcode == Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize;

        Ok(is_setup)
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> InsExecutorE1<F>
    for EcDoubleStep<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EcDoublePreCompute>()
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
        let pre_compute: &mut EcDoublePreCompute = data.borrow_mut();

        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        if is_setup {
            Ok(execute_e1_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(curve_type) = get_curve_type(pre_compute.modulus, pre_compute.a_coeff) {
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
    for EcDoubleStep<BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<EcDoublePreCompute>>()
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
        let pre_compute: &mut E2PreCompute<EcDoublePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let is_setup = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        if is_setup {
            Ok(execute_e2_setup_impl::<_, _, BLOCKS, BLOCK_SIZE>)
        } else if let Some(curve_type) =
            get_curve_type(pre_compute.data.modulus, pre_compute.data.a_coeff)
        {
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
    let pre_compute: &EcDoublePreCompute = pre_compute.borrow();
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
    let pre_compute: &EcDoublePreCompute = pre_compute.borrow();

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
    let pre_compute: &E2PreCompute<EcDoublePreCompute> = pre_compute.borrow();
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
    let pre_compute: &E2PreCompute<EcDoublePreCompute> = pre_compute.borrow();
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
    pre_compute: &EcDoublePreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read memory values for the point
    let read_data: [[u8; BLOCK_SIZE]; BLOCKS] = {
        let address = rs_vals[0];
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    };

    let output_data = match CURVE_TYPE {
        x if x == CurveType::K256 as u8 => ec_double::<0, BLOCKS, BLOCK_SIZE>(read_data),
        x if x == CurveType::P256 as u8 => ec_double::<1, BLOCKS, BLOCK_SIZE>(read_data),
        x if x == CurveType::BN254 as u8 => ec_double::<2, BLOCKS, BLOCK_SIZE>(read_data),
        x if x == CurveType::BLS12_381 as u8 => ec_double::<3, BLOCKS, BLOCK_SIZE>(read_data),
        _ => ec_double_generic::<BLOCKS, BLOCK_SIZE>(
            read_data,
            pre_compute.modulus.clone(),
            pre_compute.a_coeff.clone(),
        ),
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
    pre_compute: &EcDoublePreCompute,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read the setup input data
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

    // Extract second field element as the a coefficient
    let a_bytes: Vec<u8> = setup_input_data[BLOCKS / 2..]
        .iter()
        .flatten()
        .copied()
        .collect();
    let input_a = BigUint::from_bytes_le(&a_bytes);

    // Assert that the inputs match the expected values
    assert_eq!(
        input_prime, *pre_compute.modulus,
        "Setup: input prime must match field modulus"
    );
    assert_eq!(
        input_a, *pre_compute.a_coeff,
        "Setup: input a coefficient must match expected value"
    );

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[inline(always)]
fn ec_double_generic<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
    field_modulus: BigUint,
    a_coeff: BigUint,
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let field_element_bytes = BLOCKS * BLOCK_SIZE;
    let half_bytes = field_element_bytes / 2;

    // Extract coordinates from input data
    let x1_bytes: Vec<u8> = input_data[..BLOCKS / 2].iter().flatten().copied().collect();
    let y1_bytes: Vec<u8> = input_data[BLOCKS / 2..].iter().flatten().copied().collect();

    // Convert to BigUint for modular arithmetic
    let x1 = BigUint::from_bytes_le(&x1_bytes);
    let y1 = BigUint::from_bytes_le(&y1_bytes);

    // Elliptic curve point doubling formula:
    // lambda = (3 * x1^2 + a) / (2 * y1) mod p
    // x3 = lambda^2 - 2 * x1 mod p
    // y3 = lambda * (x1 - x3) - y1 mod p

    // Calculate lambda = (3 * x1^2 + a) / (2 * y1) mod p
    let x1_squared = (&x1 * &x1) % &field_modulus;
    let three_x1_squared = (&x1_squared * 3u32) % &field_modulus;
    let numerator = (&three_x1_squared + &a_coeff) % &field_modulus;

    let two_y1 = (&y1 * 2u32) % &field_modulus;
    let two_y1_inv = two_y1
        .modinv(&field_modulus)
        .expect("Modular inverse should exist for valid EC points");
    let lambda = (&numerator * &two_y1_inv) % &field_modulus;

    // Calculate x3 = lambda^2 - 2 * x1 mod p
    let lambda_squared = (&lambda * &lambda) % &field_modulus;
    let two_x1 = (&x1 * 2u32) % &field_modulus;
    let x3 = (&field_modulus + &lambda_squared - &two_x1) % &field_modulus;

    // Calculate y3 = lambda * (x1 - x3) - y1 mod p
    let x1_minus_x3 = if x1 >= x3 {
        (&x1 - &x3) % &field_modulus
    } else {
        (&field_modulus + &x1 - &x3) % &field_modulus
    };

    let y3 = {
        let temp = (&lambda * &x1_minus_x3) % &field_modulus;
        if temp >= y1 {
            (&temp - &y1) % &field_modulus
        } else {
            (&field_modulus + &temp - &y1) % &field_modulus
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
