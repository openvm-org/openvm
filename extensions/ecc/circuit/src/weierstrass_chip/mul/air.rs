//! AIR for the `EC_MUL` chip.
//!
//! Constrains the ladder across rows, and the memory accesses on the digest row that follows them.
//!
//! Two accumulators run together: the point `R = m*P` in the expression, and the bits `B` of the
//! signs chosen so far in [`EcMulHeaderCols::scalar_acc`]. `B` is never a multiplier, only
//! bookkeeping. An invariant ties them, and every step preserves it:
//!
//! ```text
//! m  = 2B + 1
//! m' = 2m + sigma = 2(2B + 1) + (2b - 1) = 2(2B + b) + 1 = 2B' + 1
//! ```
//!
//! It holds at the seed, where `m = 1` and `B = 0`. So checking `2B + 1` against the scalar on the
//! digest row checks that `m = k`, and hence that `R = k*P`. Wrong signs give a different `B` and
//! fail the check. Even operands fail it too, since `2B + 1` is odd for every `B`.
//!
//! Three things are assumed rather than constrained:
//!
//! - The scalar is below the curve order. The `mul` module's argument that the incomplete affine
//!   formulas never degenerate needs this. Callers enforce it, as they do for `EC_ADD_NE`.
//! - The guest called `SETUP_EC_MUL`. Under continuations only the first segment sees the setup
//!   row, so this is enforced at the program level.
//! - The base point is on the curve.

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
    ec_mul_digest_offset, ec_mul_header_width, ec_mul_width, sign_of, EcMulDigestCols,
    EcMulHeaderCols, EC_MUL_SIGN_PATTERNS, EC_MUL_STEPS_PER_ROW, IN_ACC_X, IN_ACC_Y, IN_PX, IN_PY,
    SCALAR_ACC_LIMBS, SCALAR_ACC_LIMBS_PER_BYTE, SCALAR_LIMBS,
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
        // supplies one built from the setup inputs.
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

        // This row's scalar bits, read off the one-hot sign flags instead of being stored:
        // `b_j = sum of flag_f over the patterns whose j-th sign is positive`, which is degree 1.
        // Packed most significant first, they are the accumulator limb this row contributes.
        let digits: AB::Expr = (0..EC_MUL_STEPS_PER_ROW)
            .map(|step| {
                let bit: AB::Expr = (0..EC_MUL_SIGN_PATTERNS)
                    .filter(|&pattern| sign_of(pattern, step) > 0)
                    .map(|pattern| flags[pattern].into())
                    .sum();
                bit * AB::Expr::from_u32(1 << (EC_MUL_STEPS_PER_ROW - 1 - step))
            })
            .sum();

        // `is_setup` must agree with the value `FieldExpr` derives internally,
        // `is_valid − Σflags`. On a compute row `is_valid == is_compute`.
        let flag_sum: AB::Expr = flags.iter().map(|&f| f.into()).sum();
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
        // A digest must be preceded by the ladder it summarises. The rule below is a transition
        // constraint, so it says nothing about row zero; without this, a digest there would fire
        // every memory and execution interaction with no compute predecessor at all.
        builder.when_first_row().assert_zero(local.is_digest);
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
        // The accumulator starts at `P`. The most significant digit is `+1` for every odd scalar
        // in range, so the ladder seeds itself from the point it already holds, with no memory read
        // and no sign to store.
        //
        // Gated on `!is_setup`, since a setup row's leading inputs are already pinned to the prime
        // and `a` by `FieldExpr`. Without the gate it would have to satisfy both at once.
        let first_real_compute = local.is_first_compute * (AB::Expr::ONE - local.is_setup);
        for (acc, p) in inputs[IN_ACC_X].iter().zip(&inputs[IN_PX]) {
            builder.when(first_real_compute.clone()).assert_eq(*acc, *p);
        }
        for (acc, p) in inputs[IN_ACC_Y].iter().zip(&inputs[IN_PY]) {
            builder.when(first_real_compute.clone()).assert_eq(*acc, *p);
        }
        // The bit accumulator starts empty. This also holds on setup rows, where every flag is
        // clear, so the contributed limb is zero and the accumulator stays zero.
        for &limb in local.scalar_acc.iter() {
            builder.when(local.is_first_compute).assert_zero(limb);
        }

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
        // Both stored selectors must be *defined*, not merely written by trace generation. Without
        // this, `is_real_digest` is a free column: clearing it on a real digest row disables the
        // result link and the scalar binding below while the memory and execution interactions,
        // gated by `is_digest`, still fire. A prover could then read any point and scalar, write
        // any result, and prove nothing connects them to the ladder.
        builder.assert_eq(
            local.is_real_digest,
            local.is_digest * (AB::Expr::ONE - local.is_setup),
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

        // B' = 2^EC_MUL_STEPS_PER_ROW * B + digits. With limbs sized to one row's contribution
        // this is a shift: the new low limb holds this row's digits and the rest copy a neighbour,
        // so there is nothing to carry and nothing to range check.
        when_in_instruction.assert_eq(next.scalar_acc[0], digits);
        for i in 1..SCALAR_ACC_LIMBS {
            when_in_instruction.assert_eq(next.scalar_acc[i], local.scalar_acc[i - 1]);
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
            when_both_compute.assert_eq(next_cols.inputs[IN_ACC_X][i], out_x[i]);
            when_both_compute.assert_eq(next_cols.inputs[IN_ACC_Y][i], out_y[i]);
            // The base point is constant for the whole instruction.
            when_both_compute.assert_eq(next_cols.inputs[IN_PX][i], inputs[IN_PX][i]);
            when_both_compute.assert_eq(next_cols.inputs[IN_PY][i], inputs[IN_PY][i]);
        }

        // ==== Compute → digest handoff ======================================================
        // Gated on `is_digest`, not `is_real_digest`: a setup instruction's operands need binding
        // as much as a multiplication's. On a setup row `FieldExpr` pins `inputs[IN_PX]` and
        // `inputs[IN_PY]` to the prime and `a`, so linking them to the point read is what ties the
        // memory operand to the values the setup check enforces. Only the scalar binding below is
        // genuinely setup-specific, since a setup row's scalar operand is a placeholder.
        let mut when_to_digest = builder.when_transition();
        let mut when_to_digest = when_to_digest.when(next.is_digest);

        for i in 0..NUM_LIMBS {
            // The result written to memory is the last row's output.
            when_to_digest.assert_eq(next_digest.result_x[i], out_x[i]);
            when_to_digest.assert_eq(next_digest.result_y[i], out_y[i]);
            // The point read from memory is what the rows consumed. It is constant across compute
            // rows, so linking it once here propagates to all of them.
            when_to_digest.assert_eq(next_digest.point_x[i], inputs[IN_PX][i]);
            when_to_digest.assert_eq(next_digest.point_y[i], inputs[IN_PY][i]);
        }

        // ==== Digest row ====================================================================
        // The scalar read from memory must equal `2B + 1` for the `B` the rows accumulated. See
        // the note on binding at the top of this file. Skipped for setup, whose scalar operand is a
        // placeholder.
        for &carry in local_digest.scalar_carry.iter() {
            builder.assert_bool(carry);
        }
        let mut carry_in = AB::Expr::ONE;
        for i in 0..SCALAR_LIMBS {
            // Byte `i` of `B`, from the accumulator limbs it spans, least significant first.
            let b_byte: AB::Expr = (0..SCALAR_ACC_LIMBS_PER_BYTE)
                .map(|j| {
                    local.scalar_acc[i * SCALAR_ACC_LIMBS_PER_BYTE + j]
                        * AB::Expr::from_u32(1 << (j * EC_MUL_STEPS_PER_ROW))
                })
                .sum();
            builder.when(local.is_real_digest).assert_eq(
                b_byte * AB::Expr::TWO + carry_in,
                local_digest.scalar_data[i]
                    + local_digest.scalar_carry[i] * AB::Expr::from_u32(1 << 8),
            );
            carry_in = local_digest.scalar_carry[i].into();
        }
        // No carry may leave the top byte, which pins `2B + 1 < 2^256` and so the scalar's width.
        builder
            .when(local.is_real_digest)
            .assert_zero(local_digest.scalar_carry[SCALAR_LIMBS - 1]);

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
