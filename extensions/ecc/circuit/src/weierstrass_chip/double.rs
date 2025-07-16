use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use crypto_bigint::{Encoding, U256};
use k256::{
    elliptic_curve::{
        sec1::{FromEncodedPoint, ToEncodedPoint},
        PrimeField,
    },
    AffinePoint, EncodedPoint, FieldElement, ProjectivePoint,
};
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
        MatrixRecordArena, NewVmChipWrapper, Result, StepExecutorE1, StepExecutorE2,
        VmSegmentState,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper, POINTER_MAX_BITS},
};
use openvm_circuit_derive::{
    InsExecutorE1, InsExecutorE2, InstructionExecutor, TraceFiller, TraceStep,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, Chip, ChipUsageGetter,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_CELL_BITS,
};
use openvm_mod_circuit_builder::{
    ExprBuilder, ExprBuilderConfig, FieldExpr, FieldExpressionCoreAir, FieldVariable,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::{Field, PrimeField32};

use super::{
    utils::{blocks_to_field_element, field_element_to_blocks, CurveType},
    WeierstrassAir,
};

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
#[derive(TraceStep, TraceFiller)]
pub struct EcDoubleStep<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub FieldExprVecHeapStep<1, BLOCKS, BLOCK_SIZE>,
);

#[derive(Chip, ChipUsageGetter, InstructionExecutor, InsExecutorE1, InsExecutorE2)]
pub struct EcDoubleChip<F: Field, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    pub  NewVmChipWrapper<
        F,
        WeierstrassAir<1, BLOCKS, BLOCK_SIZE>,
        EcDoubleStep<BLOCKS, BLOCK_SIZE>,
        MatrixRecordArena<F>,
    >,
);

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize>
    EcDoubleChip<F, BLOCKS, BLOCK_SIZE>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        mem_helper: SharedMemoryHelper<F>,
        pointer_max_bits: usize,
        config: ExprBuilderConfig,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker: SharedVariableRangeCheckerChip,
        a_biguint: BigUint,
    ) -> Self {
        let expr = ec_double_ne_expr(config, range_checker.bus(), a_biguint);

        let local_opcode_idx = vec![
            Rv32WeierstrassOpcode::EC_DOUBLE as usize,
            Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
        ];

        let air = WeierstrassAir::new(
            Rv32VecHeapAdapterAir::new(
                execution_bridge,
                memory_bridge,
                bitwise_lookup_chip.bus(),
                pointer_max_bits,
            ),
            FieldExpressionCoreAir::new(expr.clone(), offset, local_opcode_idx.clone(), vec![]),
        );

        let step = EcDoubleStep(FieldExprVecHeapStep::new(
            Rv32VecHeapAdapterStep::new(pointer_max_bits, bitwise_lookup_chip),
            expr,
            offset,
            local_opcode_idx,
            vec![],
            range_checker,
            "EcDouble",
            true,
        ));
        Self(NewVmChipWrapper::<_, _, _, MatrixRecordArena<_>>::new(
            air, step, mem_helper,
        ))
    }
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
            modulus: &self.0 .0.expr.builder.prime,
            a_coeff: &self.0 .0.expr.setup_values[0],
        };

        let local_opcode = opcode.local_opcode_idx(self.0 .0.offset);
        let is_setup = local_opcode == Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize;

        Ok(is_setup)
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> StepExecutorE1<F>
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
        } else {
            // Check if it's k256 in pre-compute
            let is_k256 = pre_compute.modulus
                == &BigUint::from_bytes_be(&U256::from_be_hex(FieldElement::MODULUS).to_be_bytes())
                && pre_compute.a_coeff == &BigUint::from(0u32);

            let fn_ptr = if is_k256 {
                execute_e1_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::K256 as u8 }>
            } else {
                execute_e1_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::Generic as u8 }>
            };

            Ok(fn_ptr)
        }
    }
}

impl<F: PrimeField32, const BLOCKS: usize, const BLOCK_SIZE: usize> StepExecutorE2<F>
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
        } else {
            // Check if it's k256 in pre-compute
            let is_k256 = pre_compute.data.modulus
                == &BigUint::from_bytes_be(&U256::from_be_hex(FieldElement::MODULUS).to_be_bytes())
                && pre_compute.data.a_coeff == &BigUint::from(0u32);

            let fn_ptr = if is_k256 {
                execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::K256 as u8 }>
            } else {
                execute_e2_impl::<_, _, BLOCKS, BLOCK_SIZE, { CurveType::Generic as u8 }>
            };

            Ok(fn_ptr)
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
    vm_state: &mut VmSegmentState<F, CTX>,
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
    vm_state: &mut VmSegmentState<F, CTX>,
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
    vm_state: &mut VmSegmentState<F, CTX>,
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
    vm_state: &mut VmSegmentState<F, CTX>,
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
    vm_state: &mut VmSegmentState<F, CTX>,
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

    let output_data = if CURVE_TYPE == CurveType::K256 as u8 {
        ec_double_k256::<BLOCKS, BLOCK_SIZE>(read_data)
    } else {
        ec_double_generic::<BLOCKS, BLOCK_SIZE>(
            read_data,
            pre_compute.modulus.clone(),
            pre_compute.a_coeff.clone(),
        )
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
    vm_state: &mut VmSegmentState<F, CTX>,
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
fn ec_double_k256<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    // Extract coordinates
    let x1 = blocks_to_field_element(input_data[..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element(input_data[BLOCKS / 2..].as_flattened());

    let point = EncodedPoint::from_affine_coordinates(&x1.to_bytes(), &y1.to_bytes(), false);
    let point = AffinePoint::from_encoded_point(&point).unwrap();

    let result = (ProjectivePoint::from(point).double()).to_affine();

    let encoded = result.to_encoded_point(false);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    match encoded.coordinates() {
        k256::elliptic_curve::sec1::Coordinates::Uncompressed { x, y } => {
            let x_fe = FieldElement::from_bytes(x).unwrap();
            let y_fe = FieldElement::from_bytes(y).unwrap();

            field_element_to_blocks(&x_fe, &mut output, 0);
            field_element_to_blocks(&y_fe, &mut output, BLOCKS / 2);
        }
        _ => panic!("Expected uncompressed coordinates"),
    }
    output
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
