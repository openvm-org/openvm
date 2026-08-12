//! AIR for the `EC_MUL` chip.
//!
//! Constrains the ladder across rows, and the memory accesses on the final ladder row.
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
//! final row — after completing `B` with that row's own digits — checks that `m = k`, and hence
//! that `R = k*P`. Wrong signs give a different `B` and fail the check. Even operands fail it too,
//! since `2B + 1` is odd for every `B`.
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
    ec_mul_header_width, ec_mul_io_offset, ec_mul_width, sign_of, EcMulHeaderCols, EcMulIoCols,
    EC_MUL_FINAL_ROW_IDX, EC_MUL_SIGN_PATTERNS, EC_MUL_STEPS_PER_ROW, IN_ACC_X, IN_ACC_Y, IN_PX,
    IN_PY, SCALAR_ACC_LIMBS, SCALAR_ACC_LIMBS_PER_BYTE, SCALAR_LIMBS,
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
        let io_offset = ec_mul_io_offset(expr_width);

        let main = builder.main();
        let local_row = main.row_slice(0).expect("row window should have two rows");
        let next_row = main.row_slice(1).expect("row window should have two rows");

        let local: &EcMulHeaderCols<AB::Var> = local_row[..header_width].borrow();
        let next: &EcMulHeaderCols<AB::Var> = next_row[..header_width].borrow();

        let local_expr = &local_row[header_width..io_offset];
        let next_expr = &next_row[header_width..io_offset];

        let local_io: &EcMulIoCols<AB::Var, NUM_LIMBS, BLOCKS> = local_row[io_offset..].borrow();

        // ==== Row typing ====================================================================
        builder.assert_bool(local.is_compute);
        builder.assert_bool(local.is_final);
        builder.assert_bool(local.is_first_compute);
        builder.assert_bool(local.is_setup);
        // The I/O-bearing row is a ladder row itself; padding rows carry no selector.
        builder.assert_zero(local.is_final * (AB::Expr::ONE - local.is_compute));
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
        // A continuation row extends a compute predecessor. The gate is boolean because
        // `is_first_compute` implies `is_compute`.
        builder
            .when_transition()
            .when(next.is_compute - next.is_first_compute)
            .assert_one(local.is_compute);
        // An instruction may begin only after padding or after a *completed* chain, so a fresh
        // start cannot cut a ladder short of its I/O row.
        builder
            .when_transition()
            .when(next.is_first_compute)
            .assert_zero(local.is_compute - local.is_final);
        // The final row terminates its chain: whatever follows must start fresh or be padding.
        // Without this a chain could fire its I/O at row `EC_MUL_FINAL_ROW_IDX` and keep going.
        builder
            .when_transition()
            .assert_zero(local.is_final * (next.is_compute - next.is_first_compute));

        // ==== Row index =====================================================================
        // The I/O row is pinned by value. Together with `is_first_compute` forcing index zero and
        // the increment below, an `is_final` row provably terminates a complete
        // [`EC_MUL_FINAL_ROW_IDX`]-predecessor chain. A chain that never sets `is_final` fires no
        // interaction and merely wastes its rows.
        builder.assert_zero(
            local.is_final * (local.row_idx - AB::Expr::from_usize(EC_MUL_FINAL_ROW_IDX)),
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
        // Both selectors are degree 1, which leaves room to further gate them on `!is_setup` below.
        // `continuation` is 1 exactly when `next` continues `local`'s ladder: it is boolean
        // because `is_first_compute` implies `is_compute`, and the sequencing rules above make a
        // set value mean precisely "same instruction as the previous row".
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
        // this, `is_real_final` is a free column: clearing it on a real final row disables the
        // scalar binding below while the memory and execution interactions, gated by `is_final`,
        // still fire. A prover could then read any scalar and prove nothing connects it to the
        // rows' sign flags.
        builder.assert_eq(
            local.is_real_final,
            local.is_final * (AB::Expr::ONE - local.is_setup),
        );
        let in_instruction = continuation.clone();

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
        when_in_instruction.assert_eq(next.scalar_acc[0], digits.clone());
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

        // ==== Final row =====================================================================
        // The scalar read from memory must equal `2B + 1`, where `B` is the accumulator completed
        // with this row's own digits: limb 0 is the degree-1 digit form and every other limb is
        // the entering accumulator shifted by one, exactly the shift the transition constraint
        // would apply. See the note on binding at the top of this file. Skipped for setup, whose
        // scalar operand is a placeholder.
        let completed_acc = |limb: usize| -> AB::Expr {
            if limb == 0 {
                digits.clone()
            } else {
                local.scalar_acc[limb - 1].into()
            }
        };
        for &carry in local_io.scalar_carry.iter() {
            builder.assert_bool(carry);
        }
        let mut carry_in = AB::Expr::ONE;
        for i in 0..SCALAR_LIMBS {
            // Byte `i` of the completed `B`, from the limbs it spans, least significant first.
            let b_byte: AB::Expr = (0..SCALAR_ACC_LIMBS_PER_BYTE)
                .map(|j| {
                    completed_acc(i * SCALAR_ACC_LIMBS_PER_BYTE + j)
                        * AB::Expr::from_u32(1 << (j * EC_MUL_STEPS_PER_ROW))
                })
                .sum();
            builder.when(local.is_real_final).assert_eq(
                b_byte * AB::Expr::TWO + carry_in,
                local_io.scalar_data[i] + local_io.scalar_carry[i] * AB::Expr::from_u32(1 << 8),
            );
            carry_in = local_io.scalar_carry[i].into();
        }
        // No carry may leave the top byte, which pins `2B + 1 < 2^256` and so the scalar's width.
        builder
            .when(local.is_real_final)
            .assert_zero(local_io.scalar_carry[SCALAR_LIMBS - 1]);

        self.eval_io(
            builder,
            local,
            local_io,
            [&inputs[IN_PX], &inputs[IN_PY]],
            [out_x, out_y],
        );
    }
}

impl<const NUM_LIMBS: usize, const BLOCKS: usize> EcMulAir<NUM_LIMBS, BLOCKS> {
    /// The instruction's memory accesses, all gated by `is_final`, so one `EC_MUL` costs
    /// `EC_MUL_REGISTER_READS + BLOCKS + SCALAR_BLOCKS + BLOCKS` accesses regardless of the number
    /// of ladder steps.
    ///
    /// The base point and the result have no stored copies: `point` is the final row's expression
    /// inputs, constrained constant across the instruction's rows (and pinned to `(modulus, a)` on
    /// setup rows), and `result` is that row's expression outputs. Reading and writing those
    /// columns directly is what ties the memory operands to the ladder.
    fn eval_io<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        header: &EcMulHeaderCols<AB::Var>,
        io: &EcMulIoCols<AB::Var, NUM_LIMBS, BLOCKS>,
        point: [&[AB::Var]; 2],
        result: [&[AB::Var]; 2],
    ) {
        let is_final = header.is_final;

        let start_timestamp = io.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            start_timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // ==== Register reads =================================================================
        // rs1 (point pointer), rs2 (scalar pointer), then rd, matching the order the executor
        // reads them and the convention the vec-heap adapter uses.
        for ((ptr, val), aux) in [
            (io.rs1_ptr, &io.rs1_val),
            (io.rs2_ptr, &io.rs2_val),
            (io.rd_ptr, &io.rd_val),
        ]
        .into_iter()
        .zip(io.rs_read_aux.iter())
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
                .eval(builder, is_final);

            // Bound the high u16 cell against the guest pointer limit.
            self.range_bus
                .range_check(
                    ptr_bound_from_high_u16_expr::<AB::Expr, _>(
                        val[PTR_U16_LIMBS - 1],
                        self.ptr_max_bits,
                    ),
                    U16_BITS,
                )
                .eval(builder, is_final);
        }

        let rd_addr: AB::Expr = u16_limbs_to_ptr(&io.rd_val);
        let point_addr: AB::Expr = u16_limbs_to_ptr(&io.rs1_val);
        let scalar_addr: AB::Expr = u16_limbs_to_ptr(&io.rs2_val);

        let heap = AB::F::from_u32(MEMORY_AS);

        // A point is stored as `x ‖ y`, so block `blk` spans limbs
        // `[blk * MEMORY_BLOCK_BYTES, (blk+1) * MEMORY_BLOCK_BYTES)` of that concatenation.
        let coord_block =
            |x: &[AB::Var], y: &[AB::Var], blk: usize| -> [AB::Expr; MEMORY_BLOCK_BYTES] {
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
        for (blk, aux) in io.point_read_aux.iter().enumerate() {
            let bytes = coord_block(point[0], point[1], blk);
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
                .eval(builder, is_final);
        }

        // ==== Read the scalar ===============================================================
        for (blk, aux) in io.scalar_read_aux.iter().enumerate() {
            let bytes: [AB::Expr; MEMORY_BLOCK_BYTES] =
                std::array::from_fn(|i| io.scalar_data[blk * MEMORY_BLOCK_BYTES + i].into());
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
                .eval(builder, is_final);
        }

        // ==== Write the result ==============================================================
        for (blk, aux) in io.write_aux.iter().enumerate() {
            let bytes = coord_block(result[0], result[1], blk);
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
                .eval(builder, is_final);
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
                    io.rd_ptr.into(),
                    io.rs1_ptr.into(),
                    io.rs2_ptr.into(),
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(MEMORY_AS),
                ],
                ExecutionState::new(io.from_state.pc, io.from_state.timestamp),
                AB::F::from_usize(timestamp_delta),
            )
            .eval(builder, is_final);
    }
}
