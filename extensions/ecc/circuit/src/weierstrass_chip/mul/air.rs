//! AIR constraints for multirow EC_MUL chip
//!
//! ## Constraint Overview
//!
//! The AIR defines constraints for 257 rows per EC_MUL instruction:
//!
//! ### Row Type Selectors
//! - `is_compute_row`, `is_digest_row` are boolean and mutually exclusive for valid rows
//! - `is_valid = is_compute_row + is_digest_row` (0 for padding rows)
//! - `row_idx` encoded via Encoder (22 columns, max_degree=2, reserve_invalid=true)
//!
//! ### Compute Row Constraints
//! - FieldExpr SubAir evaluation for point doubling and addition
//! - Scalar bit selection from `control.scalar_bits`
//! - `is_inf` tracking for safe denominator selection
//!
//! ### State Transitions
//! - Accumulator update: FieldExpr output → next row's `rx_limbs`, `ry_limbs`
//! - Constant propagation: scalar_bits, base_px, base_py, is_setup
//! - First compute row: accumulator starts at infinity (all zeros)
//!
//! ### Digest Row Constraints
//! - Memory I/O: register reads, scalar read, point reads/writes
//! - Execution bridge interaction
//! - Scalar bit decomposition
//! - Base point limb decomposition
//!
//! ## KNOWN ISSUE: FieldExpr Evaluated on Digest Rows
//!
//! The `SubAir::eval(&self.ec_mul_step_expr, ...)` at line ~418 is called for ALL rows,
//! including digest rows. FieldExpr reads `is_valid` from `local[base_width]`, but for
//! digest rows this contains `from_state.pc` (non-zero), causing constraint failures.
//!
//! **FIX**: Add a padding field to `EcMulDigestCols` so `local[base_width] = 0` for
//! digest rows. See `columns.rs` for details.

use std::borrow::Borrow;

use openvm_circuit::{
    arch::ExecutionBridge,
    system::{
        memory::{offline_checker::MemoryBridge, MemoryAddress},
        SystemPort,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, encoder::Encoder, utils::not,
    var_range::VariableRangeCheckerBus, SubAir,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpr};
use openvm_rv32im_circuit::adapters::abstract_compose;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::{
    ec_mul_step_expr, EcMulComputeCols, EcMulDigestCols, EcMulFlagsCols, EC_MUL_TOTAL_ROWS,
};
use crate::{EC_MUL_SCALAR_BITS, EC_MUL_SCALAR_BYTES};

/// AIR for multirow EC_MUL chip
/// NUM_LIMBS: Number of limbs for field element representation
/// NUM_BLOCKS: Number of blocks for base point and result point
/// BLOCK_SIZE: Size of each block in bytes (varies by curve: 32 for secp256k1/P-256, 16 for P-384,
/// etc.)
#[derive(Clone)]
pub struct EcMulAir<const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub ptr_max_bits: usize,

    /// Opcode offset for this curve instance
    /// offset = CLASS_OFFSET + curve_index * COUNT
    pub offset: usize,

    // FieldExpr for EC operations (stores width dynamically)
    pub ec_mul_step_expr: FieldExpr,

    // Encoder for row index
    pub row_idx_encoder: Encoder,
}

impl<const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        system_port: SystemPort,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        ptr_max_bits: usize,
        offset: usize,
        config: ExprBuilderConfig,
        range_checker_bus: VariableRangeCheckerBus,
        a_biguint: num_bigint::BigUint,
    ) -> Self {
        // Verify that NUM_LIMBS matches the FieldExpr config
        // NUM_LIMBS should equal config.num_limbs (limbs per field element)
        // For byte-sized limbs: NUM_LIMBS = ceil(field_bits / 8)
        assert_eq!(
            NUM_LIMBS, config.num_limbs,
            "NUM_LIMBS ({}) must match ExprBuilderConfig.num_limbs ({})",
            NUM_LIMBS, config.num_limbs
        );

        // Also verify BLOCK_SIZE * (NUM_BLOCKS / 2) == NUM_LIMBS for coordinate layout
        // (Each coordinate uses NUM_BLOCKS/2 blocks of BLOCK_SIZE bytes = NUM_LIMBS limbs)
        assert_eq!(
            NUM_LIMBS,
            BLOCK_SIZE * (NUM_BLOCKS / 2),
            "NUM_LIMBS ({}) must equal BLOCK_SIZE * (NUM_BLOCKS/2) = {} * {} = {}",
            NUM_LIMBS,
            BLOCK_SIZE,
            NUM_BLOCKS / 2,
            BLOCK_SIZE * (NUM_BLOCKS / 2)
        );

        Self {
            execution_bridge: ExecutionBridge::new(
                system_port.execution_bus,
                system_port.program_bus,
            ),
            memory_bridge: system_port.memory_bridge,
            bitwise_lookup_bus,
            ptr_max_bits,
            offset,
            ec_mul_step_expr: ec_mul_step_expr(config, range_checker_bus, a_biguint),
            // reserve_invalid = true: the zero point (0,...,0) is reserved for padding/invalid rows
            row_idx_encoder: Encoder::new(EC_MUL_TOTAL_ROWS, 2, true),
        }
    }
}

impl<F: Field, const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    BaseAirWithPublicValues<F> for EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
}
impl<F: Field, const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    PartitionedBaseAir<F> for EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
}

impl<F: Field, const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize> BaseAir<F>
    for EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
    fn width(&self) -> usize {
        use super::{ec_mul_compute_base_width, ec_mul_digest_width};

        // Compute row width = base columns + FieldExpr width (dynamic)
        let compute_base = ec_mul_compute_base_width::<NUM_LIMBS>();
        let compute_width = compute_base + BaseAir::<F>::width(&self.ec_mul_step_expr);

        // Digest row width
        let digest_width = ec_mul_digest_width::<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>();

        // Total width is the max of both row types
        compute_width.max(digest_width)
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_LIMBS: usize,
        const NUM_BLOCKS: usize,
        const BLOCK_SIZE: usize,
    > Air<AB> for EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
    fn eval(&self, builder: &mut AB) {
        self.eval_row_type_selectors(builder);
        self.eval_constant_propagation(builder);
        self.eval_compute_rows(builder);
        self.eval_state_transitions(builder);
        self.eval_digest_row(builder);
    }
}

impl<const NUM_LIMBS: usize, const NUM_BLOCKS: usize, const BLOCK_SIZE: usize>
    EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>
{
    /// Constrain row type flags and row index encoding
    fn eval_row_type_selectors<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        // Extract flags
        let flags_width = EcMulFlagsCols::<AB::Var>::width();
        let flags: &EcMulFlagsCols<AB::Var> = local[..flags_width].borrow();
        let next_flags: &EcMulFlagsCols<AB::Var> = next[..flags_width].borrow();

        // === All flags must be boolean ===
        builder.assert_bool(flags.is_compute_row);
        builder.assert_bool(flags.is_digest_row);
        builder.assert_bool(flags.is_first_compute_row);
        builder.assert_bool(flags.is_setup);
        builder.assert_bool(flags.is_inf);

        // === Define is_valid for gating constraints ===
        // is_valid = is_compute_row + is_digest_row
        // For valid rows: is_valid = 1 (either compute or digest)
        // For padding rows: is_valid = 0 (both are 0)
        let is_valid = flags.is_compute_row + flags.is_digest_row;
        builder.assert_bool(is_valid.clone());

        // Row index encoding - only valid for valid rows
        // (Encoder constraints are evaluated unconditionally, but that's OK as padding rows are
        // zeroed)
        self.row_idx_encoder.eval(builder, &flags.row_idx);

        // === is_first_compute_row = (row_idx == 0) AND is_compute_row ===
        let is_row_0 = self
            .row_idx_encoder
            .contains_flag::<AB>(&flags.row_idx, &[0]);
        builder.assert_eq(flags.is_first_compute_row, is_row_0 * flags.is_compute_row);

        // === is_inf constraints ===
        // is_inf = 1 on the first compute row (R starts at infinity)
        builder
            .when(flags.is_first_compute_row)
            .assert_one(flags.is_inf);

        // === Link row_idx to row type for EC_MUL instruction ===
        // row_idx = 256 implies is_digest_row = 1 (within an EC_MUL instruction)
        let is_row_256 = self
            .row_idx_encoder
            .contains_flag::<AB>(&flags.row_idx, &[256]);

        // When is_digest_row, row_idx must be 256
        builder
            .when(flags.is_digest_row)
            .assert_one(is_row_256.clone());

        // When is_compute_row, row_idx must be 0-255
        builder.when(flags.is_compute_row).assert_zero(is_row_256);

        // === Transition constraints ===
        // Within an EC_MUL instruction, row_idx increments by 1
        // compute_row -> compute_row: row_idx increments
        // compute_row (row 255) -> digest_row (row 256): row_idx increments
        let both_in_instruction =
            (flags.is_compute_row) * (next_flags.is_compute_row + next_flags.is_digest_row);

        for current_idx in 0..EC_MUL_TOTAL_ROWS - 1 {
            let is_current = self
                .row_idx_encoder
                .contains_flag::<AB>(&flags.row_idx, &[current_idx]);
            let is_next = self
                .row_idx_encoder
                .contains_flag::<AB>(&next_flags.row_idx, &[current_idx + 1]);

            builder
                .when_transition()
                .when(both_in_instruction.clone())
                .when(is_current)
                .assert_one(is_next);
        }

        // compute_row (row 255) -> digest_row
        let is_row_255 = self
            .row_idx_encoder
            .contains_flag::<AB>(&flags.row_idx, &[255]);
        builder
            .when_transition()
            .when(flags.is_compute_row)
            .when(is_row_255)
            .assert_one(next_flags.is_digest_row);

        // === Digest row must be followed by first_compute_row (start of next EC_MUL) ===
        // Only applies when the next row is valid (not a padding row)
        let next_is_valid = next_flags.is_compute_row + next_flags.is_digest_row;
        builder
            .when_transition()
            .when(flags.is_digest_row)
            .when(next_is_valid)
            .assert_one(next_flags.is_first_compute_row);
    }

    /// Constrain memory I/O operations for digest row
    fn eval_io<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        digest_cols: &EcMulDigestCols<AB::Var, NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>,
        gate: AB::Var,
        opcode: AB::Expr,
    ) {
        let timestamp = digest_cols.from_state.timestamp;
        let mut timestamp_delta = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        // === Register reads (3 reads) ===
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    digest_cols.rd_ptr,
                ),
                digest_cols.dst_ptr,
                timestamp_pp(),
                &digest_cols.rs_read_aux[0],
            )
            .eval(builder, gate);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    digest_cols.rs1_ptr,
                ),
                digest_cols.basepoint_ptr,
                timestamp_pp(),
                &digest_cols.rs_read_aux[1],
            )
            .eval(builder, gate);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    digest_cols.rs2_ptr,
                ),
                digest_cols.scalar_ptr,
                timestamp_pp(),
                &digest_cols.rs_read_aux[2],
            )
            .eval(builder, gate);

        let dst_ptr_f: AB::Expr = abstract_compose(digest_cols.dst_ptr);
        let basepoint_ptr_f: AB::Expr = abstract_compose(digest_cols.basepoint_ptr);
        let scalar_ptr_f: AB::Expr = abstract_compose(digest_cols.scalar_ptr);

        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);

        // === Point reads ===
        for (i, (block, aux)) in digest_cols
            .basepoint_data
            .iter()
            .zip(&digest_cols.reads_point_aux)
            .enumerate()
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e,
                        basepoint_ptr_f.clone() + AB::Expr::from_canonical_usize(i * BLOCK_SIZE),
                    ),
                    *block,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, gate);
        }

        // === Scalar read ===
        self.memory_bridge
            .read(
                MemoryAddress::new(e, scalar_ptr_f.clone()),
                digest_cols.scalar_data,
                timestamp_pp(),
                &digest_cols.reads_scalar_aux,
            )
            .eval(builder, gate);

        // === Write result ===
        for (i, (block, aux)) in digest_cols
            .result_data
            .iter()
            .zip(&digest_cols.writes_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e,
                        dst_ptr_f.clone() + AB::Expr::from_canonical_usize(i * BLOCK_SIZE),
                    ),
                    *block,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, gate);
        }

        // === Range check pointers ===
        let shift = AB::Expr::from_canonical_usize(
            1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.ptr_max_bits),
        );

        self.bitwise_lookup_bus
            .send_range(
                digest_cols.basepoint_ptr[RV32_REGISTER_NUM_LIMBS - 1] * shift.clone(),
                digest_cols.scalar_ptr[RV32_REGISTER_NUM_LIMBS - 1] * shift.clone(),
            )
            .eval(builder, gate);

        self.bitwise_lookup_bus
            .send_range(
                digest_cols.dst_ptr[RV32_REGISTER_NUM_LIMBS - 1] * shift,
                AB::Expr::ZERO,
            )
            .eval(builder, gate);

        // === Execution Bridge: Update VM state ===
        // timestamp_delta = 3 register reads + NUM_BLOCKS point reads + 1 scalar read + NUM_BLOCKS
        // writes
        let timestamp_delta_val = 3 + NUM_BLOCKS + 1 + NUM_BLOCKS;

        self.execution_bridge
            .execute_and_increment_pc(
                opcode,
                [
                    digest_cols.rd_ptr.into(),
                    digest_cols.rs1_ptr.into(),
                    digest_cols.rs2_ptr.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                digest_cols.from_state,
                AB::Expr::from_canonical_usize(timestamp_delta_val),
            )
            .eval(builder, gate);
    }

    /// Constrain compute rows (FieldExpr operations)
    ///
    /// ## KNOWN ISSUE
    ///
    /// This function evaluates `SubAir::eval(&self.ec_mul_step_expr, ...)` on ALL rows,
    /// including digest rows. For digest rows:
    /// - `local[base_width]` contains `from_state.pc` (non-zero)
    /// - FieldExpr interprets this as `is_valid` (should be 0 or 1)
    /// - `assert_bool(is_valid)` fails because pc is not boolean
    /// - `is_setup = is_valid - flags[0] - flags[1]` is garbage
    /// - Range check interactions sent with wrong counts
    ///
    /// **FIX**: Add a padding field to `EcMulDigestCols` before `from_state` so that
    /// `local[base_width] = 0` for digest rows. The field should never be written to,
    /// remaining 0 from trace zeroing.
    fn eval_compute_rows<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);

        // Parse base columns dynamically
        use super::ec_mul_compute_base_width;

        let base_width = ec_mul_compute_base_width::<NUM_LIMBS>();
        let compute_cols: &EcMulComputeCols<AB::Var, NUM_LIMBS> = local[..base_width].borrow();

        let is_compute = compute_cols.flags.is_compute_row;

        // === Evaluate FieldExpr (handles all EC arithmetic) ===
        // FieldExpr columns start at offset `base_width`.
        // For compute rows: these are the actual FieldExpr columns (is_valid=1, etc.)
        // For digest rows: these are digest-specific columns (from_state.pc, etc.)
        //
        // ISSUE: FieldExpr's is_valid is read from local[base_width], which for digest rows
        // contains from_state.pc (non-zero). This causes FieldExpr constraints to fail.
        let expr_width = BaseAir::<AB::F>::width(&self.ec_mul_step_expr);
        let expr_cols = &local[base_width..base_width + expr_width];

        // Gate the FieldExpr evaluation with is_compute_row
        SubAir::eval(&self.ec_mul_step_expr, builder, expr_cols);

        // Load the FieldExpr columns to access inputs, vars, and flags
        let field_cols = self.ec_mul_step_expr.load_vars(expr_cols);

        // Link FieldExpr is_valid to is_compute_row
        builder
            .when(is_compute)
            .assert_eq(field_cols.is_valid, AB::Expr::ONE);
        builder
            .when(not::<AB::Expr>(is_compute))
            .assert_zero(field_cols.is_valid);

        // === Link FieldExpr inputs to control columns ===
        // Input 0: Rx (current accumulator x)
        for i in 0..NUM_LIMBS {
            builder
                .when(is_compute)
                .assert_eq(field_cols.inputs[0][i], compute_cols.control.rx_limbs[i]);
        }

        // Input 1: Ry (current accumulator y)
        for i in 0..NUM_LIMBS {
            builder
                .when(is_compute)
                .assert_eq(field_cols.inputs[1][i], compute_cols.control.ry_limbs[i]);
        }

        // Input 2: Px (base point x)
        for i in 0..NUM_LIMBS {
            builder.when(is_compute).assert_eq(
                field_cols.inputs[2][i],
                compute_cols.control.base_px_limbs[i],
            );
        }

        // Input 3: Py (base point y)
        for i in 0..NUM_LIMBS {
            builder.when(is_compute).assert_eq(
                field_cols.inputs[3][i],
                compute_cols.control.base_py_limbs[i],
            );
        }

        // === Link FieldExpr flag to scalar bit ===
        // Extract bit index from row_idx (0..255) and map to scalar bit (MSB first: 255..0)
        let mut num_selected_row = AB::Expr::ZERO;
        let mut selected_bit = AB::Expr::ZERO;
        for row_idx in 0..256 {
            let is_this_row = self
                .row_idx_encoder
                .contains_flag::<AB>(&compute_cols.flags.row_idx, &[row_idx]);
            let bit_idx = 255 - row_idx; // MSB first
            num_selected_row += is_this_row.clone();
            selected_bit += is_this_row * compute_cols.control.scalar_bits[bit_idx].into();
        }

        // Check that the selected bit is boolean
        builder.when(is_compute).assert_one(num_selected_row);
        builder.when(is_compute).assert_bool(selected_bit.clone());

        // Link selected bit to FieldExpr flag[0]
        builder
            .when(is_compute)
            .assert_eq(field_cols.flags[0], selected_bit.clone());

        // === Link is_inf to FieldExpr flag[1] ===
        // flags[1] = is_setup OR is_inf (use safe denominators)
        // When R is at infinity (is_inf = 1), we can't compute real lambda values
        // because the denominators would be zero. Use safe denominators (1) instead.
        // Boolean OR: a OR b = a + b - a * b
        let is_setup = compute_cols.flags.is_setup.clone();
        let is_inf = compute_cols.flags.is_inf.clone();
        let use_safe_denom = is_setup.clone() + is_inf.clone() - is_setup.clone() * is_inf.clone();
        builder
            .when(is_compute)
            .assert_eq(field_cols.flags[1], use_safe_denom);

        // === Transition constraint for is_inf ===
        // is_inf_next = is_inf * (1 - bit)
        // R stays at infinity only if R was at infinity AND we didn't add P (bit = 0)
        // Once we see a bit = 1, R becomes a valid point and stays valid forever
        let next = main.row_slice(1);
        let next_cols: &EcMulComputeCols<AB::Var, NUM_LIMBS> = next[..base_width].borrow();
        let is_inf_next = compute_cols.flags.is_inf * (AB::Expr::ONE - selected_bit.clone());
        builder
            .when_transition()
            .when(is_compute)
            .when(next_cols.flags.is_compute_row)
            .assert_eq(next_cols.flags.is_inf, is_inf_next);
    }

    /// Constrain digest row (memory I/O + data decomposition)
    fn eval_digest_row<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let digest_width = super::ec_mul_digest_width::<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>();
        let digest_cols: &EcMulDigestCols<AB::Var, NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE> =
            local[..digest_width].borrow();

        let is_digest_row = digest_cols.flags.is_digest_row;

        // === Memory I/O ===
        // Compute opcode based on is_setup flag:
        // opcode = offset + (is_setup * SETUP_EC_MUL + (1 - is_setup) * EC_MUL)
        // where offset = CLASS_OFFSET + curve_index * COUNT
        let ec_mul_local = AB::Expr::from_canonical_usize(Rv32WeierstrassOpcode::EC_MUL as usize);
        let setup_ec_mul_local =
            AB::Expr::from_canonical_usize(Rv32WeierstrassOpcode::SETUP_EC_MUL as usize);
        let offset_expr = AB::Expr::from_canonical_usize(self.offset);
        let is_setup: AB::Expr = digest_cols.flags.is_setup.into();
        let local_opcode =
            is_setup.clone() * setup_ec_mul_local + (AB::Expr::ONE - is_setup) * ec_mul_local;
        let opcode = offset_expr + local_opcode;
        self.eval_io(builder, digest_cols, is_digest_row, opcode);

        // === Digest-specific constraints: Decompose scalar into bits ===
        // For each of the 256 bits, we need to:
        // 1. Constrain the bit is boolean
        // 2. Link it to digest_cols.control.scalar_bits[i]
        // 3. Constrain that the bits reconstruct the scalar_data bytes

        for byte_idx in 0..EC_MUL_SCALAR_BYTES {
            // Each byte should reconstruct from its 8 bits
            let mut reconstructed_byte = AB::Expr::ZERO;
            for bit_idx in 0..8 {
                let global_bit_idx = byte_idx * 8 + bit_idx;
                let bit = digest_cols.control.scalar_bits[global_bit_idx];

                // Constrain bit is boolean
                builder.when(is_digest_row).assert_bool(bit);

                // Add to byte reconstruction
                reconstructed_byte += AB::Expr::from_canonical_usize(1 << bit_idx) * bit;
            }

            // Constrain reconstructed byte equals the scalar data byte
            builder
                .when(is_digest_row)
                .assert_eq(reconstructed_byte, digest_cols.scalar_data[byte_idx]);
        }

        // === Digest-specific constraints: Decompose base point into limbs ===
        // Reconstruct limbs from bytes (assuming limbs are in little-endian byte order)
        let block_per_coord = NUM_BLOCKS / 2;

        for block_idx in 0..block_per_coord {
            for byte_idx in 0..BLOCK_SIZE {
                let limb_idx = block_idx * BLOCK_SIZE + byte_idx;
                let px_limb = digest_cols.basepoint_data[block_idx][byte_idx];
                let py_limb = digest_cols.basepoint_data[block_idx + block_per_coord][byte_idx];
                builder
                    .when(is_digest_row)
                    .assert_eq(px_limb, digest_cols.control.base_px_limbs[limb_idx]);
                builder
                    .when(is_digest_row)
                    .assert_eq(py_limb, digest_cols.control.base_py_limbs[limb_idx]);
            }
        }
    }

    /// Constrain state transitions between rows
    fn eval_state_transitions<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        use super::ec_mul_compute_base_width;

        let base_width = ec_mul_compute_base_width::<NUM_LIMBS>();
        let local_compute: &EcMulComputeCols<AB::Var, NUM_LIMBS> = local[..base_width].borrow();
        let next_compute: &EcMulComputeCols<AB::Var, NUM_LIMBS> = next[..base_width].borrow();

        // === Compute to compute transition ===
        let both_compute = local_compute.flags.is_compute_row * next_compute.flags.is_compute_row;

        // FieldExpr outputs become next row's accumulator
        let output_indices = self.ec_mul_step_expr.output_indices();
        let expr_width = BaseAir::<AB::F>::width(&self.ec_mul_step_expr);
        let expr_cols = &local[base_width..base_width + expr_width];
        let local_field_cols = self.ec_mul_step_expr.load_vars(expr_cols);

        // Output 0: next Rx, Output 1: next Ry
        let outputs_x = &local_field_cols.vars[output_indices[0]];
        let outputs_y = &local_field_cols.vars[output_indices[1]];

        for i in 0..NUM_LIMBS {
            builder
                .when_transition()
                .when(both_compute.clone())
                .assert_eq(next_compute.control.rx_limbs[i], outputs_x[i]);

            builder
                .when_transition()
                .when(both_compute.clone())
                .assert_eq(next_compute.control.ry_limbs[i], outputs_y[i]);
        }

        // === First compute row starts with infinity ===
        let is_first_compute = self
            .row_idx_encoder
            .contains_flag::<AB>(&local_compute.flags.row_idx, &[0]);

        // R = infinity: set accumulator to zero (or your encoding of infinity)
        for i in 0..NUM_LIMBS {
            builder
                .when(is_first_compute.clone())
                .assert_zero(local_compute.control.rx_limbs[i]);
            builder
                .when(is_first_compute.clone())
                .assert_zero(local_compute.control.ry_limbs[i]);
        }

        // === Last compute row output equals digest row write data ===
        let is_last_compute = self
            .row_idx_encoder
            .contains_flag::<AB>(&local_compute.flags.row_idx, &[EC_MUL_TOTAL_ROWS - 2]);

        // Parse the next row as digest row
        let digest_width = super::ec_mul_digest_width::<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>();
        let next_digest: &EcMulDigestCols<AB::Var, NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE> =
            next[..digest_width].borrow();

        let is_compute_to_digest = is_last_compute * next_digest.flags.is_digest_row;

        // Reconstruct result limbs from digest row's result_data blocks
        let block_per_coord = NUM_BLOCKS / 2;
        for block_idx in 0..block_per_coord {
            for byte_idx in 0..BLOCK_SIZE {
                let limb_idx = block_idx * BLOCK_SIZE + byte_idx;

                // Result X from result_data[0..block_per_coord]
                let result_x_limb = next_digest.result_data[block_idx][byte_idx];
                builder
                    .when_transition()
                    .when(is_compute_to_digest.clone())
                    .assert_eq(result_x_limb, outputs_x[limb_idx]);

                // Result Y from result_data[block_per_coord..NUM_BLOCKS]
                let result_y_limb = next_digest.result_data[block_idx + block_per_coord][byte_idx];
                builder
                    .when_transition()
                    .when(is_compute_to_digest.clone())
                    .assert_eq(result_y_limb, outputs_y[limb_idx]);
            }
        }
    }

    /// Constrain constant data propagation
    fn eval_constant_propagation<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        use super::ec_mul_compute_base_width;

        let base_width = ec_mul_compute_base_width::<NUM_LIMBS>();
        let local_cols: &EcMulComputeCols<AB::Var, NUM_LIMBS> = local[..base_width].borrow();
        let next_cols: &EcMulComputeCols<AB::Var, NUM_LIMBS> = next[..base_width].borrow();

        let not_last_row = not::<AB::Expr>(
            self.row_idx_encoder
                .contains_flag::<AB>(&local_cols.flags.row_idx, &[EC_MUL_TOTAL_ROWS - 1]),
        );

        // === is_setup flag constant within an instruction ===
        builder
            .when_transition()
            .when(not_last_row.clone())
            .assert_eq(next_cols.flags.is_setup, local_cols.flags.is_setup);

        // === Scalar bits constant ===
        for i in 0..EC_MUL_SCALAR_BITS {
            builder
                .when_transition()
                .when(not_last_row.clone())
                .assert_eq(
                    next_cols.control.scalar_bits[i],
                    local_cols.control.scalar_bits[i],
                );
        }

        // === Base point constant ===
        for i in 0..NUM_LIMBS {
            builder
                .when_transition()
                .when(not_last_row.clone())
                .assert_eq(
                    next_cols.control.base_px_limbs[i],
                    local_cols.control.base_px_limbs[i],
                );

            builder
                .when_transition()
                .when(not_last_row.clone())
                .assert_eq(
                    next_cols.control.base_py_limbs[i],
                    local_cols.control.base_py_limbs[i],
                );
        }
    }
}
