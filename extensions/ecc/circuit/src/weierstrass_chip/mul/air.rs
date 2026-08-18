use std::borrow::Borrow;

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
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
    IN_PY, SCALAR_ACC_LIMBS, SCALAR_ACC_LIMBS_PER_MEMORY_LIMB, SCALAR_MEMORY_LIMBS,
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

// No column names: the row layout embeds a runtime-sized `FieldExpr` sub-row.
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

        let local_is_valid: AB::Expr = local_expr[0].into();
        let next_is_valid: AB::Expr = next_expr[0].into();

        // ==== Row typing ====================================================================
        builder.assert_bool(local.is_final);
        builder.assert_bool(local.is_first_compute);
        // `is_final` and `is_first_compute` imply `FieldExpr::is_valid`.
        builder.assert_zero(local.is_final * (AB::Expr::ONE - local_is_valid.clone()));
        builder.assert_zero(local.is_first_compute * (AB::Expr::ONE - local_is_valid.clone()));

        SubAir::eval(&self.expr, builder, local_expr);

        let FieldExprCols {
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local_expr);
        let next_cols = self.expr.load_vars(next_expr);

        // This row's scalar bits, packed most significant first, read off the one-hot sign flags.
        let digits: AB::Expr = (0..EC_MUL_STEPS_PER_ROW)
            .map(|step| {
                let bit: AB::Expr = (0..EC_MUL_SIGN_PATTERNS)
                    .filter(|&pattern| sign_of(pattern, step) > 0)
                    .map(|pattern| flags[pattern].into())
                    .sum();
                bit * AB::Expr::from_u32(1 << (EC_MUL_STEPS_PER_ROW - 1 - step))
            })
            .sum();

        // `FieldExpr` constrains these expressions to be Boolean. A valid row with no flag is a
        // setup row; a normal row selects exactly one sign pattern.
        let local_is_setup =
            local_is_valid.clone() - flags.iter().map(|&flag| flag.into()).sum::<AB::Expr>();
        let next_is_setup = next_is_valid.clone()
            - next_cols
                .flags
                .iter()
                .map(|&flag| flag.into())
                .sum::<AB::Expr>();

        // ==== Sequencing =====================================================================
        // A compute chain must begin at `is_first_compute`.
        builder
            .when_first_row()
            .when(local_is_valid.clone())
            .assert_one(local.is_first_compute);
        // A continuation row extends a compute predecessor.
        builder
            .when_transition()
            .when(next_is_valid.clone() - next.is_first_compute)
            .assert_one(local_is_valid.clone());
        // A fresh start may follow only padding or a completed chain.
        builder
            .when_transition()
            .when(next.is_first_compute)
            .assert_zero(local_is_valid.clone() - local.is_final);
        // The final row terminates its chain.
        builder
            .when_transition()
            .assert_zero(local.is_final * (next_is_valid.clone() - next.is_first_compute));

        // ==== Row index =====================================================================
        builder.assert_zero(
            local.is_final * (local.row_idx - AB::Expr::from_usize(EC_MUL_FINAL_ROW_IDX)),
        );
        builder.assert_zero(local.is_first_compute * local.row_idx);

        // ==== First compute row ==============================================================
        // The accumulator seeds at `P`: the most significant digit of every odd in-range scalar
        // is `+1`. Gated on `!is_setup`, whose leading inputs `FieldExpr` pins to the modulus.
        let first_real_compute = local.is_first_compute * (AB::Expr::ONE - local_is_setup.clone());
        for (acc, p) in inputs[IN_ACC_X].iter().zip(&inputs[IN_PX]) {
            builder.when(first_real_compute.clone()).assert_eq(*acc, *p);
        }
        for (acc, p) in inputs[IN_ACC_Y].iter().zip(&inputs[IN_PY]) {
            builder.when(first_real_compute.clone()).assert_eq(*acc, *p);
        }
        for &limb in local.scalar_acc.iter() {
            builder.when(local.is_first_compute).assert_zero(limb);
        }

        // ==== Transitions ===================================================================
        // 1 exactly when `next` continues `local`'s ladder.
        let continuation: AB::Expr = next_is_valid.clone() - next.is_first_compute;
        // Stored so the data links below can be gated at degree 1.
        builder.assert_eq(
            local.is_ladder,
            local_is_valid.clone()
                * (AB::Expr::ONE - local_is_setup.clone())
                * (AB::Expr::ONE - local.is_first_compute),
        );
        let mut when_in_instruction = builder.when_transition();
        let mut when_in_instruction = when_in_instruction.when(continuation);

        when_in_instruction.assert_eq(next.row_idx, local.row_idx + AB::Expr::ONE);
        // `is_setup` is constant across the instruction.
        when_in_instruction.assert_eq(next_is_setup, local_is_setup.clone());
        when_in_instruction.assert_zero(next.is_first_compute);

        // B' = 2^EC_MUL_STEPS_PER_ROW * B + digits: a pure shift, nothing to carry or range check.
        when_in_instruction.assert_eq(next.scalar_acc[0], digits.clone());
        for i in 1..SCALAR_ACC_LIMBS {
            when_in_instruction.assert_eq(next.scalar_acc[i], local.scalar_acc[i - 1]);
        }

        // Setup rows re-supply their inputs each row, so they are excluded from the data links.
        let mut when_both_compute = builder.when_transition();
        let mut when_both_compute = when_both_compute.when(next.is_ladder);

        // The accumulator is threaded through the trace: the next row's inputs are this row's
        // outputs.
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
        // The scalar read from memory must equal `2B + 1`, with `B` completed by this row's own
        // digits. Skipped for setup, whose scalar operand is a placeholder.
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
        let is_real_final = local.is_final * (AB::Expr::ONE - local_is_setup.clone());
        let mut carry_in = AB::Expr::ONE;
        for i in 0..SCALAR_MEMORY_LIMBS {
            let b_limb: AB::Expr = (0..SCALAR_ACC_LIMBS_PER_MEMORY_LIMB)
                .map(|j| {
                    completed_acc(i * SCALAR_ACC_LIMBS_PER_MEMORY_LIMB + j)
                        * AB::Expr::from_u32(1 << (j * EC_MUL_STEPS_PER_ROW))
                })
                .sum();
            let carry_out = local_io
                .scalar_carry
                .get(i)
                .map(|&carry| AB::Expr::from(carry));
            builder.when(is_real_final.clone()).assert_eq(
                b_limb * AB::Expr::TWO + carry_in.clone(),
                local_io.scalar_data[i]
                    + carry_out.clone().unwrap_or(AB::Expr::ZERO) * AB::Expr::from_u32(1 << 16),
            );
            if let Some(carry_out) = carry_out {
                carry_in = carry_out;
            }
        }

        self.eval_io(
            builder,
            local,
            local_is_setup,
            local_io,
            [&inputs[IN_PX], &inputs[IN_PY]],
            [out_x, out_y],
        );
    }
}

impl<const NUM_LIMBS: usize, const BLOCKS: usize> EcMulAir<NUM_LIMBS, BLOCKS> {
    /// The instruction's memory accesses, all gated by `is_final`. The base point and result are
    /// read and written directly from the final row's expression columns, which ties the memory
    /// operands to the ladder.
    fn eval_io<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        header: &EcMulHeaderCols<AB::Var>,
        is_setup: AB::Expr,
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
        // rs1, rs2, then rd, matching the executor's read order.
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

        // A point is stored as `x ‖ y`.
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
            let data = std::array::from_fn(|i| io.scalar_data[blk * BLOCK_FE_WIDTH + i].into());
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        heap,
                        byte_ptr_to_u16_ptr::<AB>(
                            scalar_addr.clone() + AB::Expr::from_usize(blk * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    data,
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
        let opcode = (AB::Expr::ONE - is_setup.clone()) * ec_mul + is_setup * setup;

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
