//! `EC_MUL`: variable-base scalar multiplication.
//!
//! One instruction spans [`EC_MUL_TOTAL_ROWS`] rows: [`EC_MUL_COMPUTE_ROWS`] ladder rows and a
//! digest row holding every memory access. The accumulator moves between ladder rows by transition
//! constraint rather than through memory, so the instruction costs one point read, one scalar read
//! and one point write.
//!
//! The ladder is MSB-first double-and-add over signed digits, [`EC_MUL_STEPS_PER_ROW`] per row:
//!
//! ```text
//! R = P
//! for i in (0..EC_MUL_SCALAR_BITS).rev() {
//!     R = 2R + sigma_i * P          sigma_i in {+1, -1}
//! }
//! ```
//!
//! No digit is zero, so every step adds, and the addend is always `+-P`. There is no window and no
//! table of multiples. The gain over one bit per row comes from amortizing the row's fixed cost
//! over more point operations, not from doing fewer of them.
//!
//! # Why there is no case analysis
//!
//! Let `m_i` be the multiplier once digits down to `i` are consumed, so that
//! `m_i = 2*m_{i+1} + sigma_i`. Every `m_i` is odd, and `|m_i| < n` because `m_i` is a prefix of
//! `k`. So `m_i` is never zero mod `n`.
//!
//! The accumulator is therefore never the identity and no doubling denominator is zero. The group
//! has prime order, so there is no 2-torsion. The addition is exceptional only when `m_i = 0`,
//! excluded above, or `m_i = 2*sigma`, which parity rules out.
//!
//! Steps run as `(2R) + sigma*P`. See `field_expr` for why the other order does not work.
//!
//! # Preconditions
//!
//! The scalar operand must be odd and below the curve order. Both apply to the raw 256-bit value.
//!
//! Oddness comes from the encoding: `sum sigma_i 2^i` is odd for any choice of signs, so an even
//! operand has no digit assignment and no provable trace. The order bound is what the argument
//! above needs. Neither is checked here, as `EC_ADD_NE` does not check its distinct-x precondition.
//!
//! The guest wrappers handle both. They reduce mod `n` and, when the result is even, use `n - k`
//! and negate the product, since `(n - k) * P = -(k * P)`.

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

/// Signed digits per scalar, each `+1` or `-1`.
///
/// Substituting `sigma_i = 2*b_i - 1` collapses the ladder's multiplier to `2B + 1` for
/// `B = sum b_i 2^i`. The most significant digit is then `+1` for every odd scalar in range, so the
/// accumulator starts at `P` and needs no seed operand.
pub const EC_MUL_DIGITS: usize = EC_MUL_SCALAR_BITS + 1;

/// Digits consumed per compute row, each one step of `R = 2R +- P`.
///
/// Grouping steps amortizes the row's fixed overhead, since the digest region and header occupy
/// every row whatever arithmetic it does. Raising this halves the bit accumulator but doubles the
/// flag count, the sign flags being one-hot over patterns rather than one per digit.
///
/// Measured, against 398,864 cells for the four-case binary chip this replaces:
///
/// | steps | flags | rows | width | cells |
/// |---|---|---|---|---|
/// | 2 | 4 | 129 | 1839 | 237,231 |
/// | 4 | 16 | 65 | 3077 | 200,005 |
///
/// Four is 15% cheaper and was declined on legibility, not cost. Revisiting that needs only this
/// constant. Must divide 8, so accumulator limbs pack evenly into the scalar's bytes.
pub const EC_MUL_STEPS_PER_ROW: usize = 2;

/// One-hot flags per compute row, one per sign pattern of its [`EC_MUL_STEPS_PER_ROW`] digits.
///
/// `FieldExpr` derives `is_setup = is_valid - sum(flags)` and asserts it boolean, so at most one
/// flag may be set. Encoding the signs jointly satisfies that for free: each digit's sign, and each
/// scalar bit, is still a degree-1 form in these flags.
pub const EC_MUL_SIGN_PATTERNS: usize = 1 << EC_MUL_STEPS_PER_ROW;

/// Compute rows per instruction.
///
/// The most significant digit seeds the accumulator instead of being folded into it, leaving the
/// remaining [`EC_MUL_SCALAR_BITS`] digits to divide evenly among the rows.
pub const EC_MUL_COMPUTE_ROWS: usize = EC_MUL_SCALAR_BITS / EC_MUL_STEPS_PER_ROW;
/// Total trace rows per instruction: [`EC_MUL_COMPUTE_ROWS`] plus the digest row.
pub const EC_MUL_TOTAL_ROWS: usize = EC_MUL_COMPUTE_ROWS + 1;

const _: () = assert!(EC_MUL_SCALAR_BITS.is_multiple_of(EC_MUL_STEPS_PER_ROW));
const _: () = assert!(8usize.is_multiple_of(EC_MUL_STEPS_PER_ROW));

// The rvr lifter restates these values, since it cannot depend on this crate. Metered execution
// through either backend has to report the same trace height and read the same scalar width.
#[cfg(feature = "rvr")]
const _: () = {
    assert!(rvr_openvm_ext_ecc::EC_MUL_TRACE_ROWS as usize == EC_MUL_TOTAL_ROWS);
    assert!(rvr_openvm_ext_ecc::EC_MUL_SCALAR_DWORDS as usize == SCALAR_BLOCKS);
};

/// Scalar width in 8-bit limbs, matching the coordinate `limb_bits` used by the ECC chips.
pub const SCALAR_LIMBS: usize = EC_MUL_SCALAR_BITS / 8;
/// Memory blocks spanned by the scalar.
pub const SCALAR_BLOCKS: usize = SCALAR_LIMBS / openvm_circuit::arch::MEMORY_BLOCK_BYTES;

/// Width of the bit accumulator `B`, in limbs of [`EC_MUL_STEPS_PER_ROW`] bits.
///
/// A limb sized to one row's contribution makes `B = 2^EC_MUL_STEPS_PER_ROW * B + digits` a shift,
/// so there are no carries and nothing to range check. Each limb copies its predecessor's
/// neighbour, and the incoming limb is a degree-1 form in the flags, so it is already in range.
pub const SCALAR_ACC_LIMBS: usize = EC_MUL_SCALAR_BITS / EC_MUL_STEPS_PER_ROW;
/// Accumulator limbs spanned by one scalar byte.
pub const SCALAR_ACC_LIMBS_PER_BYTE: usize = 8 / EC_MUL_STEPS_PER_ROW;
