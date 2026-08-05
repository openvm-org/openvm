//! AIR for the `EC_MUL` chip.
//!
//! Constrains the ladder over [`EC_MUL_COMPUTE_ROWS`] rows and the instruction's memory accesses on
//! the digest row. Every transition constraint is gated by at most a degree-2 selector, keeping the
//! AIR's maximum constraint degree low enough not to raise the application's `log_blowup`;
//! `tests/ecmul_air.rs` asserts the bound.
//!
//! Two properties are assumed rather than constrained here:
//!
//! - That the guest called `SETUP_EC_MUL`, as for the neighbouring chips. With continuations only
//!   the first segment would observe the setup row, so it is enforced at the program level.
//! - Curve membership of the base point. The chip constrains the group law, not that `P` lies on
//!   the curve.

use std::borrow::Borrow;

use num_bigint::BigUint;
use num_traits::{One, Zero};
use openvm_circuit::{arch::ExecutionBridge, system::memory::offline_checker::MemoryBridge};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerBus, ColumnsAir, SubAir};
use openvm_mod_circuit_builder::{FieldExpr, FieldExprCols};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use super::{
    ec_mul_digest_offset, ec_mul_header_width, ec_mul_width, EcMulDigestCols, EcMulHeaderCols,
    FLAG_DBL, FLAG_DBL_ADD, FLAG_INF_STAY, FLAG_INF_TAKE, IN_PX, IN_PY, IN_RX, IN_RY, SCALAR_LIMBS,
};

/// `NUM_LIMBS` is the coordinate width in 8-bit limbs; `BLOCKS` is the memory blocks per point.
#[derive(Clone)]
pub struct EcMulAir<const NUM_LIMBS: usize, const BLOCKS: usize> {
    /// The ladder-step expression, evaluated on every compute row.
    pub expr: FieldExpr,
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    /// Maximum bit width of guest byte pointers.
    pub ptr_max_bits: usize,
    /// Global opcode offset for this curve's chip instance.
    pub offset: usize,
    /// Little-endian 8-bit limbs of `n - 1`, where `n` is the curve order. Used to bound the
    /// scalar, which is what keeps the incomplete affine addition sound — see
    /// [`EcMulDigestCols::scalar_lt_borrow`].
    order_minus_one: Vec<u8>,
}

impl<const NUM_LIMBS: usize, const BLOCKS: usize> EcMulAir<NUM_LIMBS, BLOCKS> {
    pub fn new(
        expr: FieldExpr,
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        range_bus: VariableRangeCheckerBus,
        ptr_max_bits: usize,
        offset: usize,
        curve_order: &BigUint,
    ) -> Self {
        assert!(!curve_order.is_zero(), "curve order must be nonzero");
        let mut order_minus_one = (curve_order - BigUint::one()).to_bytes_le();
        order_minus_one.resize(SCALAR_LIMBS, 0);
        Self {
            expr,
            execution_bridge,
            memory_bridge,
            range_bus,
            ptr_max_bits,
            offset,
            order_minus_one,
        }
    }

    fn expr_width<F: Field>(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field, const NUM_LIMBS: usize, const BLOCKS: usize> BaseAir<F>
    for EcMulAir<NUM_LIMBS, BLOCKS>
{
    fn width(&self) -> usize {
        ec_mul_width::<NUM_LIMBS, BLOCKS>(self.expr_width::<F>())
    }
}

// No column names provided: the row layout embeds a `FieldExpr` sub-row whose width is only known
// at runtime, so it is built dynamically rather than from a `StructReflection` struct — the same
// reason `FieldExpr` and `FieldExpressionCoreAir` have empty impls.
impl<const NUM_LIMBS: usize, const BLOCKS: usize> ColumnsAir for EcMulAir<NUM_LIMBS, BLOCKS> {}

impl<F: Field, const NUM_LIMBS: usize, const BLOCKS: usize> BaseAirWithPublicValues<F>
    for EcMulAir<NUM_LIMBS, BLOCKS>
{
}
impl<F: Field, const NUM_LIMBS: usize, const BLOCKS: usize> PartitionedBaseAir<F>
    for EcMulAir<NUM_LIMBS, BLOCKS>
{
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const BLOCKS: usize> Air<AB>
    for EcMulAir<NUM_LIMBS, BLOCKS>
{
    fn eval(&self, builder: &mut AB) {
        let expr_width = self.expr_width::<AB::F>();
        let header_width = ec_mul_header_width();
        let digest_offset = ec_mul_digest_offset(expr_width);

        let main = builder.main();
        let local_row = main.row_slice(0).expect("row window should have two rows");
        let next_row = main.row_slice(1).expect("row window should have two rows");

        let local: &EcMulHeaderCols<AB::Var> = local_row[..header_width].borrow();
        let next: &EcMulHeaderCols<AB::Var> = next_row[..header_width].borrow();

        let local_expr = &local_row[header_width..digest_offset];
        let next_expr = &next_row[header_width..digest_offset];

        let local_digest: &EcMulDigestCols<AB::Var, NUM_LIMBS, BLOCKS> =
            local_row[digest_offset..].borrow();
        let next_digest: &EcMulDigestCols<AB::Var, NUM_LIMBS, BLOCKS> =
            next_row[digest_offset..].borrow();

        // ==== Row typing ====================================================================
        builder.assert_bool(local.is_compute);
        builder.assert_bool(local.is_digest);
        builder.assert_bool(local.is_first_compute);
        builder.assert_bool(local.is_setup);
        // A row is either a ladder step, the digest, or padding — never two of those.
        builder.assert_bool(local.is_compute + local.is_digest);
        // `is_first_compute` implies `is_compute`.
        builder.assert_zero(local.is_first_compute * (AB::Expr::ONE - local.is_compute));

        // The expression is active exactly on compute rows, leaving its region all-zero elsewhere,
        // which is what satisfies its ungated constraints on digest and padding rows.
        builder.assert_eq(local_expr[0], local.is_compute);

        // Evaluate the ladder step. `FieldExpr` reads `is_valid` from `local_expr[0]`.
        SubAir::eval(&self.expr, builder, local_expr);

        let FieldExprCols {
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local_expr);
        let next_cols = self.expr.load_vars(next_expr);

        let f_dbl = flags[FLAG_DBL];
        let f_dbl_add = flags[FLAG_DBL_ADD];
        let f_inf_stay = flags[FLAG_INF_STAY];
        let f_inf_take = flags[FLAG_INF_TAKE];

        // The scalar bit and the infinity indicator are recovered from the one-hot case flags
        // rather than stored.
        let bit = f_dbl_add + f_inf_take;

        // `is_setup` must agree with the value `FieldExpr` derives internally,
        // `is_valid − Σflags`. On a compute row `is_valid == is_compute`.
        let flag_sum = f_dbl + f_dbl_add + f_inf_stay + f_inf_take;
        builder
            .when(local.is_compute)
            .assert_eq(local.is_setup, local.is_compute - flag_sum);

        // ==== Sequencing =====================================================================
        // A compute chain must begin at `is_first_compute`, which is the only constraint pinning
        // the initial accumulator and `scalar_acc` to zero. Without this a ladder could start
        // part-way through with both chosen freely.
        builder
            .when_first_row()
            .when(local.is_compute)
            .assert_one(local.is_first_compute);
        // A compute row is either the start of an instruction or a continuation of the previous
        // row. Summing also forbids both at once, so an instruction cannot begin directly after a
        // ladder row without an intervening digest row.
        builder
            .when_transition()
            .when(next.is_compute)
            .assert_one(next.is_first_compute + local.is_compute);

        // ==== Row index =====================================================================
        // The digest row is pinned by value, so the counter cannot drift.
        builder.when(local.is_digest).assert_eq(
            local.row_idx,
            AB::Expr::from_usize(super::EC_MUL_DIGEST_ROW_IDX),
        );
        // The first compute row is index 0.
        builder.assert_zero(local.is_first_compute * local.row_idx);

        // ==== First compute row ==============================================================
        // The accumulator starts at the affine identity sentinel (0, 0).
        //
        // Gated on `!is_setup` because `FieldExpr`'s setup check requires `inputs[0..]` to equal
        // the prime limbs followed by the setup values, and `inputs[0..2]` are
        // `IN_RX`/`IN_RY`. Without the gate a setup row would have to satisfy both
        // conditions at once.
        let first_real_compute = local.is_first_compute * (AB::Expr::ONE - local.is_setup);
        for (rx, ry) in inputs[IN_RX].iter().zip(&inputs[IN_RY]) {
            builder.when(first_real_compute.clone()).assert_zero(*rx);
            builder.when(first_real_compute.clone()).assert_zero(*ry);
        }
        // ...so the case must be one of the two infinity cases. On a setup row every flag is
        // clear, and this holds trivially.
        builder
            .when(local.is_first_compute)
            .assert_zero(f_dbl + f_dbl_add);
        // The scalar accumulator starts empty. This also holds on setup rows, where every flag is
        // clear so `bit` is always zero and the accumulator stays zero.
        for &limb in local.scalar_acc.iter() {
            builder.when(local.is_first_compute).assert_zero(limb);
        }

        // ==== Running scalar: s' = 2·s + bit ================================================
        // Carries are boolean because 2·s[i] + c ≤ 511.
        for &carry in local.scalar_carry.iter() {
            builder.assert_bool(carry);
        }
        // Nothing may carry out of the top limb: after `i` steps s < 2^i ≤ 2^256.
        builder
            .when(local.is_compute)
            .assert_zero(local.scalar_carry[SCALAR_LIMBS - 1]);

        // ==== Transitions ===================================================================
        // Both selectors are degree 2, so the constraints they gate stay at degree 3.
        let both_compute = local.is_compute * next.is_compute;
        let to_digest = local.is_compute * next.is_digest;
        // "next row belongs to the same instruction"
        let in_instruction = both_compute.clone() + to_digest.clone();

        let mut when_in_instruction = builder.when_transition();
        let mut when_in_instruction = when_in_instruction.when(in_instruction);

        // row_idx increments.
        when_in_instruction.assert_eq(next.row_idx, local.row_idx + AB::Expr::ONE);
        // NOTE: the two constraints below live outside this block; see `eval`'s sequencing
        // section. They are what force a compute chain to *begin* at `is_first_compute`.
        // `is_setup` is constant across the instruction.
        when_in_instruction.assert_eq(next.is_setup, local.is_setup);
        // A continuation row is never a first row.
        when_in_instruction.assert_zero(next.is_first_compute);

        // s' = 2·s + bit, limb by limb, little-endian. The incoming carry of limb 0 is the bit.
        let mut carry_in = bit.clone();
        for i in 0..SCALAR_LIMBS {
            when_in_instruction.assert_eq(
                local.scalar_acc[i] * AB::Expr::TWO + carry_in,
                next.scalar_acc[i] + local.scalar_carry[i] * AB::Expr::from_u32(1 << 8),
            );
            carry_in = local.scalar_carry[i].into();
        }

        let mut when_both_compute = builder.when_transition();
        let mut when_both_compute = when_both_compute.when(both_compute);

        // The accumulator is carried through the trace rather than memory: the next row's
        // accumulator inputs are this row's outputs.
        let out_x = &vars[self.expr.program().output_indices()[0]];
        let out_y = &vars[self.expr.program().output_indices()[1]];
        for i in 0..NUM_LIMBS {
            when_both_compute.assert_eq(next_cols.inputs[IN_RX][i], out_x[i]);
            when_both_compute.assert_eq(next_cols.inputs[IN_RY][i], out_y[i]);
            // The base point is constant for the whole instruction.
            when_both_compute.assert_eq(next_cols.inputs[IN_PX][i], inputs[IN_PX][i]);
            when_both_compute.assert_eq(next_cols.inputs[IN_PY][i], inputs[IN_PY][i]);
        }

        // is_inf' = is_inf AND NOT bit, which under the one-hot encoding is exactly `f_inf_stay`.
        let next_is_inf = next_cols.flags[FLAG_INF_STAY] + next_cols.flags[FLAG_INF_TAKE];
        when_both_compute.assert_eq(next_is_inf, f_inf_stay);

        // ==== Compute → digest handoff ======================================================
        let mut when_to_digest = builder.when_transition();
        let mut when_to_digest = when_to_digest.when(to_digest);

        for i in 0..NUM_LIMBS {
            // The result written to memory is the last step's output.
            when_to_digest.assert_eq(next_digest.result_x[i], out_x[i]);
            when_to_digest.assert_eq(next_digest.result_y[i], out_y[i]);
            // The point read from memory is the base point the ladder used. `P` is constant
            // across compute rows, so linking it once here propagates to all of them.
            when_to_digest.assert_eq(next_digest.point_x[i], inputs[IN_PX][i]);
            when_to_digest.assert_eq(next_digest.point_y[i], inputs[IN_PY][i]);
        }

        // ==== Digest row ====================================================================
        // The accumulated scalar must equal the scalar read from memory. This is also what pins the
        // per-row case flags: any row implying the wrong bit changes `scalar_acc` and fails here.
        // Skipped for setup, whose scalar operand is a dummy.
        let check_scalar = local.is_digest * (AB::Expr::ONE - local.is_setup);
        for i in 0..SCALAR_LIMBS {
            builder
                .when(check_scalar.clone())
                .assert_eq(local.scalar_acc[i], local_digest.scalar_data[i]);
        }

        let _ = (local_digest, next_digest);
    }
}
