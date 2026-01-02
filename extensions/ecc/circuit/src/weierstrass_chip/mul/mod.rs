//! Multirow EC_MUL Chip
//!
//! This module implements elliptic curve scalar multiplication using a double-and-add
//! algorithm. Each EC_MUL instruction generates 257 trace rows:
//! - 256 compute rows (one per scalar bit)
//! - 1 digest row (memory I/O)
//!
//! ## Module Structure
//!
//! - `mod.rs` (this file): Constants, type aliases, and chip constructors
//! - `columns.rs`: Column struct definitions
//! - `air.rs`: AIR constraints
//! - `field_expr.rs`: FieldExpr for EC point operations
//! - `execution.rs`: Instruction execution logic
//! - `trace.rs`: Trace generation (Executor and Filler)
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Create executor for instruction execution
//! let executor = get_ec_mul_step::<NUM_BLOCKS, BLOCK_SIZE>(...);
//!
//! // Create AIR for constraint system
//! let air = get_ec_mul_air::<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>(...);
//!
//! // Create chip (wraps filler with memory helper)
//! let chip = get_ec_mul_chip::<F, NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>(...);
//! ```
//!
//! ## Current Status
//!
//! **Working**: Execution, trace generation, most AIR constraints
//!
//! **Known Issues**:
//! 1. FieldExpr constraints fail on digest rows due to column layout mismatch (CRITICAL)
//! 2. Point addition to infinity may not be handled correctly in FieldExpr (POTENTIAL)
//! 3. Adding two equal points (P = D) causes division by zero (POTENTIAL)
//! 4. Requires `--app-log-blowup 3` due to high constraint degree (7) from Encoder (CONFIG)
//!
//! See `README.md` for details and suggested fixes.

use num_bigint::BigUint;
use openvm_circuit::{
    arch::VmChipWrapper,
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpr};

// Execution logic for EC_MUL
mod execution;

// Multirow EC_MUL implementation
mod air;
mod columns;
mod field_expr;
mod trace;

pub use air::*;
pub use columns::*;
pub use field_expr::*;
pub use trace::*;

// ========== Constants for EC_MUL ==========

pub const EC_MUL_SCALAR_BITS: usize = 256;
pub const EC_MUL_SCALAR_BYTES: usize = 32;
// Note: BLOCK_SIZE is a const generic parameter based on the curve
// It's typically 32 bytes for 256-bit curves (secp256k1, P-256)
// but can be 16 bytes for 381-bit curves with more blocks, etc.
pub const EC_MUL_COMPUTE_ROWS: usize = 256;
pub const EC_MUL_TOTAL_ROWS: usize = 257; // 256 compute + 1 digest

const EC_MUL_REGISTER_READS: usize = 3; // rd, rs1, rs2

// ========== EC_MUL Executor ==========

/// EC_MUL executor
/// NUM_BLOCKS = number of memory blocks for point I/O (typically 2 for x,y coordinates)
/// BLOCK_SIZE = bytes per memory block (typically equals FIELD_BYTES)
#[derive(Clone, derive_new::new)]
pub struct EcMulExecutor<const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub pointer_max_bits: usize,
    pub expr: FieldExpr,
    pub offset: usize,
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
    /// Scalar field modulus (curve order) for setup validation
    pub scalar_biguint: BigUint,
}

/// EC_MUL chip type alias
pub type EcMulChip<F, const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmChipWrapper<F, EcMulFiller<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>>;

// ========== EC_MUL Chip Constructors ==========

/// Get EC_MUL step executor
pub fn get_ec_mul_step<const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>(
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
    offset: usize,
    a_biguint: BigUint,
    scalar_biguint: BigUint,
) -> EcMulExecutor<NUM_BLOCKS, BLOCK_SIZE> {
    let expr = ec_mul_step_expr(config, range_checker_bus, a_biguint);
    let local_opcode_idx = vec![
        Rv32WeierstrassOpcode::EC_MUL as usize,
        Rv32WeierstrassOpcode::SETUP_EC_MUL as usize,
    ];
    EcMulExecutor::new(
        pointer_max_bits,
        expr,
        offset,
        local_opcode_idx,
        vec![],
        scalar_biguint,
    )
}

/// Get EC_MUL AIR
#[allow(clippy::too_many_arguments)]
pub fn get_ec_mul_air<const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>(
    system_port: SystemPort,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    ptr_max_bits: usize,
    offset: usize,
    config: ExprBuilderConfig,
    range_checker_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE> {
    EcMulAir::new(
        system_port,
        bitwise_lookup_bus,
        ptr_max_bits,
        offset,
        config,
        range_checker_bus,
        a_biguint,
    )
}

/// Get EC_MUL chip
#[allow(clippy::too_many_arguments)]
pub fn get_ec_mul_chip<
    F,
    const NUM_LIMBS: usize,
    const NUM_BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    config: ExprBuilderConfig,
    mem_helper: SharedMemoryHelper<F>,
    range_checker: SharedVariableRangeCheckerChip,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pointer_max_bits: usize,
    a_biguint: BigUint,
    scalar_biguint: BigUint,
) -> EcMulChip<F, NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE> {
    let expr = ec_mul_step_expr(config, range_checker.bus(), a_biguint);
    let local_opcode_idx = vec![
        Rv32WeierstrassOpcode::EC_MUL as usize,
        Rv32WeierstrassOpcode::SETUP_EC_MUL as usize,
    ];
    let filler = EcMulFiller {
        ec_mul_step_expr: expr,
        local_opcode_idx,
        opcode_flag_idx: vec![],
        range_checker,
        bitwise_lookup_chip,
        // Encoder configuration for row index (0..256):
        // - num_flags = 257 (EC_MUL_TOTAL_ROWS)
        // - max_degree = 2 (requires 22 columns, since C(24,2) = 276 >= 257)
        // - reserve_invalid = true: the zero point (0,...,0) is reserved for padding rows
        //
        // This allows encoding row indices 0-256, with padding rows having all-zero row_idx.
        row_idx_encoder: Encoder::new(EC_MUL_TOTAL_ROWS, 2, true),
        pointer_max_bits,
        should_finalize: false,
        scalar_biguint,
    };
    EcMulChip::new(filler, mem_helper)
}
