//! `EC_MUL`: variable-base scalar multiplication.
//!
//! One instruction spans a fixed number of ladder rows, the last of which carries every
//! memory access. The accumulator moves between rows by transition constraint rather than through
//! memory, so the instruction costs one point read, one scalar read and one point write.
//!
//! The ladder is MSB-first double-and-add over signed digits, a few per row:
//!
//! ```text
//! R = P
//! for i in (0..EC_MUL_SCALAR_BITS).rev() {
//!     R = 2R + sigma_i * P          sigma_i in {+1, -1}
//! }
//! ```
//!
//! No digit is zero, so every step adds `+-P`; there is no window and no table of multiples.
//!
//! # Totality
//!
//! For an odd scalar below the prime order `n`, every intermediate multiplier `m_i` is an odd
//! prefix with `0 < |m_i| < n`, so the accumulator is never the identity and the doubling never
//! degenerates. The addition `2R + sigma*P` is exceptional only for the prefix `+-(n - 1)/2`,
//! which is reachable exactly when it is odd, i.e. when `n = 3 (mod 4)`. The chip therefore
//! requires `n = 1 (mod 4)`, asserted at registration.
//!
//! # Preconditions
//!
//! The scalar operand must be odd and below the curve order, and the base point must lie on the
//! curve in the prime-order subgroup. None of these are checked here, as `EC_ADD_NE` does not
//! check its distinct-x precondition; the guest wrappers discharge them by reducing mod `n` and
//! substituting `n - k` (with a negated product) for even results.

mod air;
mod columns;
mod execution;
mod field_expr;
mod trace;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
mod cuda;
pub use air::*;
pub use columns::*;
#[cfg(feature = "cuda")]
pub(crate) use cuda::*;
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

/// Asserts that the ladder's totality argument covers a curve with the given scalar order; see
/// the module documentation.
pub fn assert_supported_scalar_order(scalar_order: &BigUint) {
    assert_eq!(
        scalar_order % 4u32,
        BigUint::from(1u32),
        "EC_MUL requires a scalar order congruent to 1 mod 4; this curve's is {scalar_order}"
    );
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
) -> EcMulAir<NUM_LIMBS, BLOCKS> {
    let expr = ec_mul_step_expr(config, range_checker_bus, a);
    EcMulAir::new(
        expr,
        exec_bridge,
        mem_bridge,
        range_checker_bus,
        ptr_max_bits,
        offset,
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
) -> EcMulChip<F, NUM_LIMBS, BLOCKS> {
    let expr = ec_mul_step_expr(config, range_checker.bus(), a);
    EcMulChip::new(expr, range_checker, mem_helper, ptr_max_bits)
}

/// Scalar width in bits.
pub const EC_MUL_SCALAR_BITS: usize = 256;

/// Digits consumed per compute row, each one step of `R = 2R +- P`. Must divide 8, so accumulator
/// limbs pack evenly into the scalar's bytes.
pub const EC_MUL_STEPS_PER_ROW: usize = 2;

/// One-hot flags per compute row, one per sign pattern of its digits.
pub const EC_MUL_SIGN_PATTERNS: usize = 1 << EC_MUL_STEPS_PER_ROW;

/// Trace rows per instruction. The most significant digit seeds the accumulator, leaving the
/// remaining digits to divide evenly among the rows.
pub const EC_MUL_COMPUTE_ROWS: usize = EC_MUL_SCALAR_BITS / EC_MUL_STEPS_PER_ROW;

const _: () = assert!(EC_MUL_SCALAR_BITS.is_multiple_of(EC_MUL_STEPS_PER_ROW));
const _: () = assert!(8usize.is_multiple_of(EC_MUL_STEPS_PER_ROW));

// The rvr lifter restates these values, since it cannot depend on this crate.
#[cfg(feature = "rvr")]
const _: () = {
    assert!(rvr_openvm_ext_ecc::EC_MUL_TRACE_ROWS as usize == EC_MUL_COMPUTE_ROWS);
    assert!(rvr_openvm_ext_ecc::EC_MUL_SCALAR_DWORDS as usize == SCALAR_BLOCKS);
};

/// Scalar width in 8-bit limbs, matching the coordinate `limb_bits` used by the ECC chips.
pub const SCALAR_LIMBS: usize = EC_MUL_SCALAR_BITS / 8;
/// Memory blocks spanned by the scalar.
pub const SCALAR_BLOCKS: usize = SCALAR_LIMBS / openvm_circuit::arch::MEMORY_BLOCK_BYTES;

/// Width of the bit accumulator, in limbs of one row's digits.
pub const SCALAR_ACC_LIMBS: usize = EC_MUL_SCALAR_BITS / EC_MUL_STEPS_PER_ROW;
/// Accumulator limbs spanned by one scalar byte.
pub const SCALAR_ACC_LIMBS_PER_BYTE: usize = 8 / EC_MUL_STEPS_PER_ROW;
