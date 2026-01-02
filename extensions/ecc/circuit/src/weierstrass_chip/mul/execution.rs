//! Execution layer for the EC_MUL multirow chip.
//!
//! This module implements the `InterpreterExecutor` and `InterpreterMeteredExecutor` traits
//! for `EcMulExecutor`. It handles the VM-side computation before trace generation.
//!
//! ## Execution Flow
//!
//! 1. **Pre-compute Phase**: Parse instruction, validate format, determine if setup mode
//! 2. **Dispatch Phase**: Identify curve type and select appropriate handler
//! 3. **Execution Phase**: Read inputs, compute EC multiplication, write outputs
//!
//! ## Setup vs Normal Mode
//!
//! - **EC_MUL**: Performs native EC multiplication using curve libraries (k256, p256, etc.)
//! - **SETUP_EC_MUL**: Validates curve parameters and runs FieldExpr 256 times for consistency
//!
//! ## Curve Type Dispatch
//!
//! The `dispatch!` macro creates monomorphized handlers for each supported curve:
//! - K256 (secp256k1)
//! - P256 (NIST P-256)
//! - BN254
//! - BLS12_381

use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use num_bigint::BigUint;
use openvm_circuit::{
    arch::*,
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_mod_circuit_builder::FieldExpr;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{EcMulExecutor, EC_MUL_SCALAR_BITS, EC_MUL_TOTAL_ROWS};
use crate::weierstrass_chip::curves::{ec_mul, get_curve_type, CurveType};

/// Run FieldExpr 256 times for EC_MUL setup.
/// This computes the same result that the trace filler will produce,
/// ensuring consistency between execution output and AIR constraints.
///
/// For setup mode, is_setup_flag = true, which uses safe denominators
/// to avoid division by zero (since inputs are curve parameters, not valid points).
fn run_ec_mul_expr_for_setup<const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>(
    expr: &FieldExpr,
    point_data: [[u8; BLOCK_SIZE]; NUM_BLOCKS],
    scalar_data: [u8; 32],
) -> [[u8; BLOCK_SIZE]; NUM_BLOCKS] {
    let blocks_per_coord = NUM_BLOCKS / 2;

    // Extract Px, Py from point_data (used as base point in each iteration)
    let px_bytes: Vec<u8> = point_data[..blocks_per_coord]
        .iter()
        .flat_map(|b| b.iter().copied())
        .collect();
    let py_bytes: Vec<u8> = point_data[blocks_per_coord..]
        .iter()
        .flat_map(|b| b.iter().copied())
        .collect();
    let px = BigUint::from_bytes_le(&px_bytes);
    let py = BigUint::from_bytes_le(&py_bytes);

    // Start with accumulator at (0, 0) - represents point at infinity
    let mut rx = BigUint::ZERO;
    let mut ry = BigUint::ZERO;

    // Convert scalar to bits (MSB first for double-and-add)
    let scalar_bits: Vec<bool> = (0..EC_MUL_SCALAR_BITS)
        .map(|i| {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            (scalar_data[byte_idx] >> bit_idx) & 1 == 1
        })
        .rev() // MSB first
        .collect();

    // Run 256 iterations of the FieldExpr
    for bit in scalar_bits {
        // Prepare inputs: [Rx, Ry, Px, Py]
        let inputs = vec![rx.clone(), ry.clone(), px.clone(), py.clone()];

        // Flags: [bit_flag, is_setup_flag]
        // bit_flag = current scalar bit
        // is_setup_flag = true (uses safe denominators)
        let flags = vec![bit, true];

        // Execute FieldExpr
        let vars = expr.execute(inputs, flags);

        // Extract outputs: [Rx_next, Ry_next]
        let output_indices = expr.output_indices();
        rx = vars[output_indices[0]].clone();
        ry = vars[output_indices[1]].clone();
    }

    // Convert final (rx, ry) back to point_data format
    let mut result = [[0u8; BLOCK_SIZE]; NUM_BLOCKS];

    // Rx into first half
    let rx_bytes = rx.to_bytes_le();
    for (i, block) in result[..blocks_per_coord].iter_mut().enumerate() {
        let start = i * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(rx_bytes.len());
        if start < rx_bytes.len() {
            block[..end - start].copy_from_slice(&rx_bytes[start..end]);
        }
    }

    // Ry into second half
    let ry_bytes = ry.to_bytes_le();
    for (i, block) in result[blocks_per_coord..].iter_mut().enumerate() {
        let start = i * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(ry_bytes.len());
        if start < ry_bytes.len() {
            block[..end - start].copy_from_slice(&ry_bytes[start..end]);
        }
    }

    result
}

/// Pre-computed data extracted from instruction during the pre-compute phase.
/// This struct is stored in the pre-compute buffer and passed to the execution handler.
///
/// The pre-compute phase runs once per instruction type during program loading,
/// while the execution handler runs for each instruction instance during VM execution.
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct EcMulPreCompute<'a> {
    /// Reference to the FieldExpr that defines EC point arithmetic constraints.
    /// Used for setup mode to run the same computation as the trace filler.
    expr: &'a FieldExpr,
    /// Expected curve order (scalar field modulus) for setup validation.
    scalar_biguint: &'a BigUint,
    /// Register addresses: [rs1 (base point pointer), rs2 (scalar pointer)]
    rs_addrs: [u8; 2],
    /// Destination register address (rd - result pointer)
    a: u8,
    /// FieldExpr flag index for multi-operation chips (unused for EC_MUL single-op)
    flag_idx: u8,
}

impl<'a, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> EcMulExecutor<NUM_BLOCKS, BLOCK_SIZE> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EcMulPreCompute<'a>,
    ) -> Result<bool, StaticProgramError> {
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
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.offset);

        // Pre-compute flag_idx
        let needs_setup = self.expr.needs_setup();
        let mut flag_idx = self.expr.num_flags() as u8;
        if needs_setup {
            // Find which opcode this is in our local_opcode_idx list
            if let Some(opcode_position) = self
                .local_opcode_idx
                .iter()
                .position(|&idx| idx == local_opcode)
            {
                // If this is NOT the last opcode (setup), get the corresponding flag_idx
                if opcode_position < self.opcode_flag_idx.len() {
                    flag_idx = self.opcode_flag_idx[opcode_position] as u8;
                }
            }
        }

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = EcMulPreCompute {
            expr: &self.expr,
            scalar_biguint: &self.scalar_biguint,
            rs_addrs,
            a: a as u8,
            flag_idx,
        };

        let is_setup = local_opcode == Rv32WeierstrassOpcode::SETUP_EC_MUL as usize;

        Ok(is_setup)
    }
}

/// Dispatch macro for curve-type-specific handler selection.
///
/// This macro creates monomorphized handler functions for each (curve_type, is_setup) combination.
/// Monomorphization enables the compiler to inline and optimize the hot path for each curve.
///
/// ## Handler Selection Logic
///
/// 1. Attempt to identify curve from (modulus, a_coeff) via `get_curve_type()`
/// 2. If recognized: select handler with specific CURVE_TYPE const generic
/// 3. If unknown + setup mode: use u8::MAX as CURVE_TYPE (runs FieldExpr)
/// 4. If unknown + normal mode: return error (unsupported curve)
macro_rules! dispatch {
    ($execute_impl:ident, $pre_compute:ident, $is_setup:ident) => {
        // Identify curve type from modulus and 'a' coefficient
        if let Some(curve_type) = {
            let modulus = &$pre_compute.expr.builder.prime;
            let a_coeff = &$pre_compute.expr.setup_values[0];
            get_curve_type(modulus, a_coeff)
        } {
            match ($is_setup, curve_type) {
                (true, CurveType::K256) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::K256 as u8 },
                    true,
                >),
                (true, CurveType::P256) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::P256 as u8 },
                    true,
                >),
                (true, CurveType::BN254) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::BN254 as u8 },
                    true,
                >),
                (true, CurveType::BLS12_381) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::BLS12_381 as u8 },
                    true,
                >),
                (false, CurveType::K256) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::K256 as u8 },
                    false,
                >),
                (false, CurveType::P256) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::P256 as u8 },
                    false,
                >),
                (false, CurveType::BN254) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::BN254 as u8 },
                    false,
                >),
                (false, CurveType::BLS12_381) => Ok($execute_impl::<
                    _,
                    _,
                    NUM_BLOCKS,
                    BLOCK_SIZE,
                    { CurveType::BLS12_381 as u8 },
                    false,
                >),
            }
        } else if $is_setup {
            Ok($execute_impl::<_, _, NUM_BLOCKS, BLOCK_SIZE, { u8::MAX }, true>)
        } else {
            Ok($execute_impl::<_, _, NUM_BLOCKS, BLOCK_SIZE, { u8::MAX }, false>)
        }
    };
}

impl<F: PrimeField32, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> InterpreterExecutor<F>
    for EcMulExecutor<NUM_BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EcMulPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut EcMulPreCompute = data.borrow_mut();
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_handler, pre_compute, is_setup)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut EcMulPreCompute = data.borrow_mut();
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_handler, pre_compute, is_setup)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> AotExecutor<F>
    for EcMulExecutor<NUM_BLOCKS, BLOCK_SIZE>
{
}

impl<F: PrimeField32, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    InterpreterMeteredExecutor<F> for EcMulExecutor<NUM_BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<EcMulPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<EcMulPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;
        dispatch!(execute_e2_handler, pre_compute_pure, is_setup)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<EcMulPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;
        dispatch!(execute_e2_handler, pre_compute_pure, is_setup)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> AotMeteredExecutor<F>
    for EcMulExecutor<NUM_BLOCKS, BLOCK_SIZE>
{
}

/// Scalar is always 32 bytes (256 bits)
const EC_MUL_SCALAR_SIZE: usize = 32;

/// Core execution implementation shared between E1 (non-metered) and E2 (metered) paths.
///
/// ## Const Generic Parameters
///
/// - `NUM_BLOCKS`: Number of memory blocks per point coordinate (e.g., 2 for 64-byte coords)
/// - `BLOCK_SIZE`: Size of each memory block in bytes (e.g., 32)
/// - `CURVE_TYPE`: Curve identifier (from CurveType enum) or u8::MAX for unknown
/// - `IS_SETUP`: true for SETUP_EC_MUL, false for EC_MUL
///
/// ## Memory Layout
///
/// - `rs1` points to base point: [Px (NUM_BLOCKS/2 blocks), Py (NUM_BLOCKS/2 blocks)]
/// - `rs2` points to scalar: 32 bytes
/// - `rd` points to result: [Rx (NUM_BLOCKS/2 blocks), Ry (NUM_BLOCKS/2 blocks)]
///
/// ## Setup Mode Memory Layout
///
/// For SETUP_EC_MUL, rs1 points to curve parameters instead:
/// - First half: field prime modulus
/// - Second half: 'a' coefficient
/// - rs2 contains: curve order (scalar field modulus)
#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const NUM_BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &EcMulPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();

    // Step 1: Read register values to get memory pointers
    // rs_vals[0] = base point address (or curve params for setup)
    // rs_vals[1] = scalar address
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Step 2: Read memory values
    // Point data: NUM_BLOCKS blocks of BLOCK_SIZE bytes each
    // For a 256-bit curve with 32-byte blocks: 2 blocks for x, 2 blocks for y = 4 total
    let point_data: [[u8; BLOCK_SIZE]; NUM_BLOCKS] =
        from_fn(|i| exec_state.vm_read(RV32_MEMORY_AS, rs_vals[0] + (i * BLOCK_SIZE) as u32));
    // Scalar: always 32 bytes (256 bits)
    let scalar_data: [u8; EC_MUL_SCALAR_SIZE] = exec_state.vm_read(RV32_MEMORY_AS, rs_vals[1]);

    // Step 3: Setup mode validation
    if IS_SETUP {
        // For setup, point_data contains: [prime (first half), coeff_a (second half)]
        // Validate the prime matches our expected field modulus
        let input_prime = BigUint::from_bytes_le(point_data[..NUM_BLOCKS / 2].as_flattened());
        if input_prime != pre_compute.expr.builder.prime {
            let err = ExecutionError::Fail {
                pc,
                msg: "EcMul: mismatched prime",
            };
            return Err(err);
        }

        // Validate the 'a' coefficient matches
        let input_a = BigUint::from_bytes_le(point_data[NUM_BLOCKS / 2..].as_flattened());
        let coeff_a = &pre_compute.expr.setup_values[0];
        if input_a != *coeff_a {
            let err = ExecutionError::Fail {
                pc,
                msg: "EcMul: mismatched coeff_a",
            };
            return Err(err);
        }

        // Validate the scalar field modulus (curve order) matches
        let input_scalar = BigUint::from_bytes_le(&scalar_data);
        if input_scalar != *pre_compute.scalar_biguint {
            let err = ExecutionError::Fail {
                pc,
                msg: "EcMul: mismatched scalar modulus",
            };
            return Err(err);
        }
    }

    // Step 4: Compute output based on mode
    let output_data = if IS_SETUP {
        // For setup: run FieldExpr 256 times to get consistent output
        // This ensures execution output matches what trace filler produces.
        // The trace filler will also run FieldExpr, so we must produce identical results.
        run_ec_mul_expr_for_setup::<NUM_BLOCKS, BLOCK_SIZE>(
            pre_compute.expr,
            point_data,
            scalar_data,
        )
    } else if CURVE_TYPE == u8::MAX {
        // Unknown curve in non-setup mode is not supported.
        // All curves should be recognized by get_curve_type() based on (modulus, a_coeff).
        // If we reach here, the curve was not properly configured.
        return Err(ExecutionError::Fail {
            pc,
            msg: "EcMul: unsupported curve (unknown curve type)",
        });
    } else {
        // For known curves: perform native EC multiplication using optimized libraries.
        // This uses k256, p256, or halo2curves depending on CURVE_TYPE.
        // The result is the point k*P where k is the scalar and P is the base point.
        ec_mul::<CURVE_TYPE, NUM_BLOCKS, BLOCK_SIZE>(point_data, scalar_data)
    };

    // Step 5: Read destination pointer from rd register
    let rd_val = u32::from_le_bytes(exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * NUM_BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    // Step 6: Write output data to memory (NUM_BLOCKS blocks)
    for (i, block) in output_data.into_iter().enumerate() {
        exec_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    // Step 7: Advance program counter
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const NUM_BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &EcMulPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<EcMulPreCompute>()).borrow();
    execute_e12_impl::<_, _, NUM_BLOCKS, BLOCK_SIZE, CURVE_TYPE, IS_SETUP>(pre_compute, exec_state)
}

/// Metered execution handler (E2 path).
///
/// This is used when trace height metering is enabled. The key difference from E1 is
/// the `on_height_change` call which reports that this instruction generates 257 rows.
///
/// This allows the VM to properly size the trace matrix before trace generation.
#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const NUM_BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let e2_pre_compute: &E2PreCompute<EcMulPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<EcMulPreCompute>>())
            .borrow();
    // Report height change: this instruction generates EC_MUL_TOTAL_ROWS (257) rows
    // This is critical for multirow chips - the VM uses this to allocate trace space
    exec_state
        .ctx
        .on_height_change(e2_pre_compute.chip_idx as usize, EC_MUL_TOTAL_ROWS as u32);
    execute_e12_impl::<_, _, NUM_BLOCKS, BLOCK_SIZE, CURVE_TYPE, IS_SETUP>(
        &e2_pre_compute.data,
        exec_state,
    )
}
