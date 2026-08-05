//! AIR for the `EC_MUL` chip.
//!
//! Constrains the ladder over [`EC_MUL_COMPUTE_ROWS`] rows and the instruction's memory accesses on
//! the digest row.
//!
//! Transition constraints are gated by degree-1 selectors, which keeps the AIR at the same maximum
//! constraint degree as the neighbouring chips. That bound matters beyond this chip: `log_blowup`
//! is derived from the configuration's `max_constraint_degree` and applies to every AIR in the
//! application, so exceeding it here would raise the proving cost of all of them.
//! `tests/ecmul_air.rs` asserts it.
//!
//! Two of those selectors are stored as columns ([`EcMulHeaderCols::is_ladder`] and
//! [`EcMulHeaderCols::is_real_digest`]) rather than formed inline, because the conjunctions they
//! stand for are degree 2 and would push the constraints they gate over the bound.
//!
//! Three properties are assumed rather than constrained here:
//!
//! - That the scalar operand is less than the curve order. The ladder's addition uses the
//!   incomplete affine formula, and at or above the order an intermediate `2R` can equal `P`,
//!   collapsing the addition constraint to `0 = 0`. Callers must enforce the bound, as they must
//!   for `EC_ADD_NE`'s distinct-x precondition.
//! - That the guest called `SETUP_EC_MUL`, as for the neighbouring chips. With continuations only
//!   the first segment would observe the setup row, so it is enforced at the program level.
//! - Curve membership of the base point. The chip constrains the group law, not that `P` lies on
//!   the curve.

use std::borrow::Borrow;

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, MEMORY_BLOCK_BYTES},
    system::memory::{
        offline_checker::{pack_u8_block, MemoryBridge},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerBus, ColumnsAir, SubAir, U16_BITS};
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{
    riscv::{MEMORY_AS, REGISTER_AS},
    LocalOpcode,
};
use openvm_mod_circuit_builder::{FieldExpr, FieldExprCols};
use openvm_riscv_circuit::adapters::{
    byte_ptr_to_u16_ptr, expand_to_block, ptr_bound_from_high_u16_expr, u16_limbs_to_ptr,
    PTR_U16_LIMBS,
};
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
}

impl<const NUM_LIMBS: usize, const BLOCKS: usize> EcMulAir<NUM_LIMBS, BLOCKS> {
    pub fn new(
        expr: FieldExpr,
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        range_bus: VariableRangeCheckerBus,
        ptr_max_bits: usize,
        offset: usize,
    ) -> Self {
        Self {
            expr,
            execution_bridge,
            memory_bridge,
            range_bus,
            ptr_max_bits,
            offset,
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

        // The expression is active exactly on compute rows. Elsewhere its region still has to hold
        // a consistent witness, since several of its constraints are ungated; trace generation
        // supplies one for zero inputs.
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
        // A digest row always follows a ladder row. Without this the digest row's data would be
        // unconstrained whenever it was not preceded by one, while its memory writes still fired.
        builder
            .when_transition()
            .when(next.is_digest)
            .assert_one(local.is_compute);

        // Both selectors are degree 1, which leaves room to further gate them on `!is_setup` below.
        // `continuation` is 1 exactly when `next` continues `local`'s ladder: when `next` is a
        // compute row the sequencing constraint forces `next.is_first_compute + local.is_compute`
        // to be 1, so subtracting the first-row flag leaves `local.is_compute`.
        let continuation: AB::Expr = next.is_compute - next.is_first_compute;
        // is_ladder = is_compute AND NOT is_setup AND NOT is_first_compute, so it can gate the data
        // links at degree 1.
        builder.assert_eq(
            local.is_ladder,
            local.is_compute
                * (AB::Expr::ONE - local.is_setup)
                * (AB::Expr::ONE - local.is_first_compute),
        );
        let to_digest: AB::Expr = next.is_digest.into();
        let in_instruction = continuation.clone() + to_digest.clone();

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

        // Setup rows are excluded from the data links below: every setup row must carry the prime
        // and the setup values in the accumulator inputs, so those inputs are re-supplied each row
        // rather than threaded from the previous one.
        let mut when_both_compute = builder.when_transition();
        let mut when_both_compute = when_both_compute.when(next.is_ladder);

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
        let mut when_to_digest = when_to_digest.when(next.is_real_digest);

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
        for i in 0..SCALAR_LIMBS {
            builder
                .when(local.is_real_digest)
                .assert_eq(local.scalar_acc[i], local_digest.scalar_data[i]);
        }

        let _ = next_digest;
        self.eval_io(builder, local, local_digest);
    }
}

impl<const NUM_LIMBS: usize, const BLOCKS: usize> EcMulAir<NUM_LIMBS, BLOCKS> {
    /// The instruction's memory accesses, all gated by `is_digest`, so one `EC_MUL` costs
    /// `EC_MUL_REGISTER_READS + BLOCKS + SCALAR_BLOCKS + BLOCKS` accesses regardless of the number
    /// of ladder steps.
    fn eval_io<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        header: &EcMulHeaderCols<AB::Var>,
        digest: &EcMulDigestCols<AB::Var, NUM_LIMBS, BLOCKS>,
    ) {
        let is_digest = header.is_digest;

        let start_timestamp = digest.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            start_timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // ==== Register reads =================================================================
        // rs1 (point pointer), rs2 (scalar pointer), then rd, matching the order the executor
        // reads them and the convention the vec-heap adapter uses.
        for ((ptr, val), aux) in [
            (digest.rs1_ptr, &digest.rs1_val),
            (digest.rs2_ptr, &digest.rs2_val),
            (digest.rd_ptr, &digest.rd_val),
        ]
        .into_iter()
        .zip(digest.rs_read_aux.iter())
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::F::from_u32(REGISTER_AS),
                        byte_ptr_to_u16_ptr::<AB>(ptr),
                    ),
                    expand_to_block(val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, is_digest);

            // Bound the high u16 cell against the guest pointer limit.
            self.range_bus
                .range_check(
                    ptr_bound_from_high_u16_expr::<AB::Expr, _>(
                        val[PTR_U16_LIMBS - 1],
                        self.ptr_max_bits,
                    ),
                    U16_BITS,
                )
                .eval(builder, is_digest);
        }

        let rd_addr: AB::Expr = u16_limbs_to_ptr(&digest.rd_val);
        let point_addr: AB::Expr = u16_limbs_to_ptr(&digest.rs1_val);
        let scalar_addr: AB::Expr = u16_limbs_to_ptr(&digest.rs2_val);

        let heap = AB::F::from_u32(MEMORY_AS);

        // A point is stored as `x ‖ y`, so block `blk` spans limbs
        // `[blk * MEMORY_BLOCK_BYTES, (blk+1) * MEMORY_BLOCK_BYTES)` of that concatenation.
        let coord_block = |x: &[AB::Var; NUM_LIMBS],
                           y: &[AB::Var; NUM_LIMBS],
                           blk: usize|
         -> [AB::Expr; MEMORY_BLOCK_BYTES] {
            std::array::from_fn(|i| {
                let idx = blk * MEMORY_BLOCK_BYTES + i;
                if idx < NUM_LIMBS {
                    x[idx].into()
                } else {
                    y[idx - NUM_LIMBS].into()
                }
            })
        };

        // ==== Read the base point ===========================================================
        for (blk, aux) in digest.point_read_aux.iter().enumerate() {
            let bytes = coord_block(&digest.point_x, &digest.point_y, blk);
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        heap,
                        byte_ptr_to_u16_ptr::<AB>(
                            point_addr.clone() + AB::Expr::from_usize(blk * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    pack_u8_block::<AB>(&bytes),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, is_digest);
        }

        // ==== Read the scalar ===============================================================
        for (blk, aux) in digest.scalar_read_aux.iter().enumerate() {
            let bytes: [AB::Expr; MEMORY_BLOCK_BYTES] =
                std::array::from_fn(|i| digest.scalar_data[blk * MEMORY_BLOCK_BYTES + i].into());
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        heap,
                        byte_ptr_to_u16_ptr::<AB>(
                            scalar_addr.clone() + AB::Expr::from_usize(blk * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    pack_u8_block::<AB>(&bytes),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, is_digest);
        }

        // ==== Write the result ==============================================================
        for (blk, aux) in digest.write_aux.iter().enumerate() {
            let bytes = coord_block(&digest.result_x, &digest.result_y, blk);
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        heap,
                        byte_ptr_to_u16_ptr::<AB>(
                            rd_addr.clone() + AB::Expr::from_usize(blk * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    pack_u8_block::<AB>(&bytes),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, is_digest);
        }

        // ==== Execution bus =================================================================
        // The opcode is selected by `is_setup`, so one chip instance serves both opcodes.
        let ec_mul = AB::Expr::from_usize(WeierstrassOpcode::EC_MUL.local_usize() + self.offset);
        let setup =
            AB::Expr::from_usize(WeierstrassOpcode::SETUP_EC_MUL.local_usize() + self.offset);
        let opcode = (AB::Expr::ONE - header.is_setup) * ec_mul + header.is_setup * setup;

        self.execution_bridge
            .execute_and_increment_pc(
                opcode,
                [
                    digest.rd_ptr.into(),
                    digest.rs1_ptr.into(),
                    digest.rs2_ptr.into(),
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(MEMORY_AS),
                ],
                ExecutionState::new(digest.from_state.pc, digest.from_state.timestamp),
                AB::F::from_usize(timestamp_delta),
            )
            .eval(builder, is_digest);
    }
}
