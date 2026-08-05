//! `EC_MUL`: variable-base scalar multiplication.
//!
//! One instruction spans [`EC_MUL_TOTAL_ROWS`] rows: [`EC_MUL_COMPUTE_ROWS`] ladder steps followed
//! by a digest row holding the instruction's memory accesses. The accumulator is carried between
//! ladder rows by transition constraints rather than through memory, so a whole multiplication
//! costs the register reads, one point read, one scalar read and one point write of a single
//! instruction.
//!
//! # Algorithm
//!
//! MSB-first binary double-and-add, one scalar bit per compute row:
//!
//! ```text
//! R = O
//! for i in (0..256).rev() {
//!     R = 2R
//!     if bit(i) { R = R + P }
//! }
//! ```
//!
//! A windowed ladder would need the `2^w` precomputed multiples of `P` addressable from within a
//! row, requiring either the whole table as columns or a per-row memory read. Binary needs only
//! `P`.
//!
//! # Preconditions
//!
//! The scalar operand must be less than the curve order, which is constrained on the digest row
//! (see [`EcMulDigestCols::scalar_lt_borrow`]). This is required for soundness rather than only
//! correctness: the ladder's addition uses the incomplete affine formula, and for a scalar at or
//! above the order an intermediate `2R` can equal `P`, at which point the constraint
//! `lambda * (Px - Dx) = Py - Dy` degenerates to `0 = 0` and leaves `lambda` unconstrained.
//!
//! The bound applies to the raw 256-bit operand; an unreduced scalar is rejected rather than
//! reduced.

mod air;
mod columns;
mod execution;
mod field_expr;
mod trace;

#[cfg(test)]
mod tests;

pub use air::*;
pub use columns::*;
pub use field_expr::*;
use num_bigint::BigUint;
use openvm_mod_circuit_builder::FieldExpressionProgram;
pub use trace::*;

/// Preflight/interpreter executor for `EC_MUL` and `SETUP_EC_MUL`.
///
/// `BLOCKS` is the number of memory blocks spanned by one point.
#[derive(Clone)]
pub struct EcMulExecutor<const BLOCKS: usize> {
    pub program: FieldExpressionProgram,
    /// Global opcode offset for this curve's chip instance.
    pub offset: usize,
}

impl<const BLOCKS: usize> EcMulExecutor<BLOCKS> {
    pub fn new(program: FieldExpressionProgram, offset: usize) -> Self {
        Self { program, offset }
    }
}

/// Constructors mirroring `get_ec_addne_*` / `get_ec_double_*`.
#[allow(clippy::too_many_arguments)]
pub fn get_ec_mul_air<const NUM_LIMBS: usize, const BLOCKS: usize>(
    exec_bridge: openvm_circuit::arch::ExecutionBridge,
    mem_bridge: openvm_circuit::system::memory::offline_checker::MemoryBridge,
    config: openvm_mod_circuit_builder::ExprBuilderConfig,
    range_checker_bus: openvm_circuit_primitives::var_range::VariableRangeCheckerBus,
    ptr_max_bits: usize,
    offset: usize,
    a: BigUint,
    curve_order: &BigUint,
) -> EcMulAir<NUM_LIMBS, BLOCKS> {
    let expr = ec_mul_step_expr(config, range_checker_bus, a);
    EcMulAir::new(
        expr,
        exec_bridge,
        mem_bridge,
        range_checker_bus,
        ptr_max_bits,
        offset,
        curve_order,
    )
}

pub fn get_ec_mul_executor<const BLOCKS: usize>(
    config: openvm_mod_circuit_builder::ExprBuilderConfig,
    range_max_bits: usize,
    offset: usize,
    a: BigUint,
) -> EcMulExecutor<BLOCKS> {
    EcMulExecutor::new(ec_mul_step_program(config, range_max_bits, a), offset)
}

pub fn get_ec_mul_chip<F, const NUM_LIMBS: usize, const BLOCKS: usize>(
    config: openvm_mod_circuit_builder::ExprBuilderConfig,
    mem_helper: openvm_circuit::system::memory::SharedMemoryHelper<F>,
    range_checker: openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip,
    ptr_max_bits: usize,
    a: BigUint,
    curve_order: &BigUint,
) -> EcMulChip<F, NUM_LIMBS, BLOCKS> {
    let expr = ec_mul_step_expr(config, range_checker.bus(), a);
    EcMulChip::new(expr, range_checker, mem_helper, ptr_max_bits, curve_order)
}

/// Scalar width in bits; one compute row per bit.
pub const EC_MUL_SCALAR_BITS: usize = 256;
/// Compute rows per instruction, one per scalar bit.
pub const EC_MUL_COMPUTE_ROWS: usize = EC_MUL_SCALAR_BITS;
/// Total trace rows per instruction: [`EC_MUL_COMPUTE_ROWS`] plus the digest row.
pub const EC_MUL_TOTAL_ROWS: usize = EC_MUL_COMPUTE_ROWS + 1;

/// Scalar width in 8-bit limbs, matching the coordinate `limb_bits` used by the ECC chips.
pub const SCALAR_LIMBS: usize = EC_MUL_SCALAR_BITS / 8;
/// Memory blocks spanned by the scalar.
pub const SCALAR_BLOCKS: usize = SCALAR_LIMBS / openvm_circuit::arch::MEMORY_BLOCK_BYTES;
