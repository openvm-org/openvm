//! Trace generation for the `EC_MUL` chip.
//!
//! Each selected postflight step becomes [`EC_MUL_TOTAL_ROWS`] consecutive rows: the ladder steps,
//! produced by evaluating the step [`FieldExpr`] once per scalar bit and carrying the accumulator
//! forward, followed by the digest row holding the memory witnesses.
//!
//! Padding rows carry a consistent witness for zero inputs with `is_valid` cleared, rather than
//! being all-zero: the expression folds the curve's `a` coefficient in as a constant, so on a zero
//! row its lambda constraint evaluates to `-a` and the ungated carry recurrences are unsatisfiable
//! whenever `a != 0`.

use std::{
    borrow::BorrowMut,
    sync::{atomic::Ordering, Arc},
};

use num_bigint::BigUint;
use openvm_circuit::{
    arch::{Postflight, PostflightError, PostflightStep, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerChip},
    TraceSubRowGenerator,
};
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS},
    VmOpcode,
};
use openvm_mod_circuit_builder::FieldExpr;
use openvm_riscv_circuit::adapters::{ptr_bound_from_ptr, ptr_to_field_u16_limbs};
use openvm_stark_backend::{
    p3_air::BaseAir, p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
};

use super::{
    ec_mul_digest_offset, ec_mul_header_width, ec_mul_width, setup_row_inputs, EcMulDigestCols,
    EcMulHeaderCols, EC_MUL_COMPUTE_ROWS, EC_MUL_DIGEST_ROW_IDX, EC_MUL_REGISTER_READS,
    EC_MUL_SCALAR_BITS, EC_MUL_TOTAL_ROWS, FLAG_DBL, FLAG_DBL_ADD, FLAG_INF_STAY, FLAG_INF_TAKE,
    NUM_STEP_FLAGS, SCALAR_BLOCKS, SCALAR_LIMBS,
};

/// Prover-side state for one curve's `EC_MUL` chip.
///
/// Not a `VmChipWrapper`: the row layout is header / expression / digest rather than adapter /
/// core, so the vec-heap adapter's filler does not apply.
/// `ChipInventory::add_executor_chip_with_tracegen` accepts any `'static` type.
pub struct EcMulChip<F, const NUM_LIMBS: usize, const BLOCKS: usize> {
    pub expr: FieldExpr,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub mem_helper: SharedMemoryHelper<F>,
    pub ptr_max_bits: usize,
}

impl<F, const NUM_LIMBS: usize, const BLOCKS: usize> EcMulChip<F, NUM_LIMBS, BLOCKS> {
    pub fn new(
        expr: FieldExpr,
        range_checker: SharedVariableRangeCheckerChip,
        mem_helper: SharedMemoryHelper<F>,
        ptr_max_bits: usize,
    ) -> Self {
        Self {
            expr,
            range_checker,
            mem_helper,
            ptr_max_bits,
        }
    }
}

/// One instruction's replayed values, before field encoding.
///
/// The layout is the ABI shared with the GPU gather kernel, so it is `repr(C)` with every `u32`
/// field ahead of every `u16` array. That ordering leaves no interior padding, which lets both
/// sides assert the same size; see [`ec_mul_trace_input_bytes`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EcMulTraceInput<const BLOCKS: usize> {
    from_pc: u32,
    from_timestamp: u32,
    /// Nonzero for `SETUP_EC_MUL`. An integer rather than a `bool` so the ABI is unambiguous.
    is_setup: u32,
    /// `[rs1, rs2, rd]`, in the AIR's timestamp order.
    reg_ptrs: [u32; EC_MUL_REGISTER_READS],
    reg_vals: [u32; EC_MUL_REGISTER_READS],
    reg_prev_timestamps: [u32; EC_MUL_REGISTER_READS],
    point_prev_timestamps: [u32; BLOCKS],
    scalar_prev_timestamps: [u32; SCALAR_BLOCKS],
    write_prev_timestamps: [u32; BLOCKS],
    point_blocks: [[u16; BLOCK_FE_WIDTH]; BLOCKS],
    scalar_blocks: [[u16; BLOCK_FE_WIDTH]; SCALAR_BLOCKS],
    write_blocks: [[u16; BLOCK_FE_WIDTH]; BLOCKS],
    write_predecessors: [[u16; BLOCK_FE_WIDTH]; BLOCKS],
}

#[cfg(feature = "cuda")]
impl<const BLOCKS: usize> EcMulTraceInput<BLOCKS> {
    /// The instruction's start timestamp, used to order projections gathered per opcode.
    pub(crate) fn start_timestamp(&self) -> u32 {
        self.from_timestamp
    }
}

/// Byte size of [`EcMulTraceInput`], stated independently of the struct so a layout change fails to
/// compile on both sides of the FFI rather than silently reinterpreting device memory.
pub(crate) const fn ec_mul_trace_input_bytes(blocks: usize) -> usize {
    let words = 3 + 3 * EC_MUL_REGISTER_READS + blocks + SCALAR_BLOCKS + blocks;
    let cells = (blocks + SCALAR_BLOCKS + 2 * blocks) * BLOCK_FE_WIDTH;
    4 * words + 2 * cells
}

const _: () = {
    use crate::{ECC_BLOCKS_32, ECC_BLOCKS_48};
    assert!(size_of::<EcMulTraceInput<ECC_BLOCKS_32>>() == ec_mul_trace_input_bytes(ECC_BLOCKS_32));
    assert!(size_of::<EcMulTraceInput<ECC_BLOCKS_48>>() == ec_mul_trace_input_bytes(ECC_BLOCKS_48));
};

fn checked_u16_pointer(byte_pointer: u32, what: &str) -> Result<u32, PostflightError> {
    if byte_pointer & 1 != 0 {
        return Err(PostflightError::new(format!(
            "{what} byte pointer {byte_pointer:#x} is not u16-aligned"
        )));
    }
    Ok(byte_pointer >> 1)
}

fn pointer_from_register(
    value: [u16; BLOCK_FE_WIDTH],
    ptr_max_bits: usize,
) -> Result<u32, PostflightError> {
    if value[2] != 0 || value[3] != 0 {
        return Err(PostflightError::new(
            "EC_MUL pointer register has nonzero upper 32 bits",
        ));
    }
    let pointer = u32::from(value[0]) | (u32::from(value[1]) << 16);
    if ptr_max_bits > u32::BITS as usize || u64::from(pointer) >= (1u64 << ptr_max_bits) {
        return Err(PostflightError::new(format!(
            "EC_MUL pointer {pointer:#x} exceeds {ptr_max_bits}-bit address space"
        )));
    }
    Ok(pointer)
}

fn add_byte_offset(base: u32, block: usize, ptr_max_bits: usize) -> Result<u32, PostflightError> {
    let offset = u32::try_from(block * MEMORY_BLOCK_BYTES)
        .map_err(|_| PostflightError::new("EC_MUL block offset overflow"))?;
    let pointer = base
        .checked_add(offset)
        .ok_or_else(|| PostflightError::new("EC_MUL byte pointer overflow"))?;
    if ptr_max_bits > u32::BITS as usize || u64::from(pointer) >= (1u64 << ptr_max_bits) {
        return Err(PostflightError::new(format!(
            "EC_MUL byte pointer {pointer:#x} exceeds {ptr_max_bits}-bit address space"
        )));
    }
    Ok(pointer)
}

/// Replays a step's memory accesses in the order the AIR assigns timestamps: `rs1`, `rs2`, `rd`,
/// the point blocks, the scalar blocks, then the result writes.
fn project_step<F: PrimeField32, const BLOCKS: usize>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    is_setup: bool,
    ptr_max_bits: usize,
) -> Result<EcMulTraceInput<BLOCKS>, PostflightError> {
    let instruction = postflight.instruction(step);
    if instruction.d.as_canonical_u32() != REGISTER_AS
        || instruction.e.as_canonical_u32() != MEMORY_AS
    {
        return Err(PostflightError::new(
            "EC_MUL instruction uses invalid address spaces",
        ));
    }

    let reg_ptrs = [
        instruction.b.as_canonical_u32(),
        instruction.c.as_canonical_u32(),
        instruction.a.as_canonical_u32(),
    ];
    let mut replay = postflight.replay(step);

    let mut reg_vals = [0u32; EC_MUL_REGISTER_READS];
    let mut reg_prev_timestamps = [0u32; EC_MUL_REGISTER_READS];
    for i in 0..EC_MUL_REGISTER_READS {
        let access = replay.read_u16(
            REGISTER_AS,
            checked_u16_pointer(reg_ptrs[i], "EC_MUL register")?,
        )?;
        reg_vals[i] = pointer_from_register(access.value, ptr_max_bits)?;
        reg_prev_timestamps[i] = access.previous_timestamp;
    }
    let [point_val, scalar_val, rd_val] = reg_vals;

    let mut point_blocks = [[0u16; BLOCK_FE_WIDTH]; BLOCKS];
    let mut point_prev_timestamps = [0u32; BLOCKS];
    for block in 0..BLOCKS {
        let byte_pointer = add_byte_offset(point_val, block, ptr_max_bits)?;
        let access = replay.read_u16(
            MEMORY_AS,
            checked_u16_pointer(byte_pointer, "EC_MUL point read")?,
        )?;
        point_blocks[block] = access.value;
        point_prev_timestamps[block] = access.previous_timestamp;
    }

    let mut scalar_blocks = [[0u16; BLOCK_FE_WIDTH]; SCALAR_BLOCKS];
    let mut scalar_prev_timestamps = [0u32; SCALAR_BLOCKS];
    for block in 0..SCALAR_BLOCKS {
        let byte_pointer = add_byte_offset(scalar_val, block, ptr_max_bits)?;
        let access = replay.read_u16(
            MEMORY_AS,
            checked_u16_pointer(byte_pointer, "EC_MUL scalar read")?,
        )?;
        scalar_blocks[block] = access.value;
        scalar_prev_timestamps[block] = access.previous_timestamp;
    }

    let mut write_blocks = [[0u16; BLOCK_FE_WIDTH]; BLOCKS];
    let mut write_predecessors = [[0u16; BLOCK_FE_WIDTH]; BLOCKS];
    let mut write_prev_timestamps = [0u32; BLOCKS];
    for block in 0..BLOCKS {
        let byte_pointer = add_byte_offset(rd_val, block, ptr_max_bits)?;
        let access = replay.write_observed_u16(
            MEMORY_AS,
            checked_u16_pointer(byte_pointer, "EC_MUL result write")?,
        )?;
        write_blocks[block] = access.value;
        write_predecessors[block] = access.previous_value;
        write_prev_timestamps[block] = access.previous_timestamp;
    }

    let next_pc = postflight
        .pc(step)
        .checked_add(DEFAULT_PC_STEP)
        .ok_or_else(|| PostflightError::new("EC_MUL next PC overflow"))?;
    replay.finish(next_pc)?;

    Ok(EcMulTraceInput {
        from_pc: postflight.pc(step),
        from_timestamp: postflight.timestamp(step),
        is_setup: u32::from(is_setup),
        reg_ptrs,
        reg_vals,
        reg_prev_timestamps,
        point_prev_timestamps,
        scalar_prev_timestamps,
        write_prev_timestamps,
        point_blocks,
        scalar_blocks,
        write_blocks,
        write_predecessors,
    })
}

fn blocks_to_bytes<const N: usize>(blocks: &[[u16; BLOCK_FE_WIDTH]; N]) -> Vec<u8> {
    let mut out = Vec::with_capacity(N * MEMORY_BLOCK_BYTES);
    for block in blocks {
        for limb in block {
            out.extend_from_slice(&limb.to_le_bytes());
        }
    }
    out
}

/// Fills one instruction's [`EC_MUL_TOTAL_ROWS`] rows.
#[allow(clippy::too_many_arguments)]
fn fill_instruction<F: PrimeField32, const NUM_LIMBS: usize, const BLOCKS: usize>(
    expr: &FieldExpr,
    range_checker: &VariableRangeCheckerChip,
    mem_helper: &openvm_circuit::system::memory::MemoryAuxColsFactory<F>,
    ptr_max_bits: usize,
    dummy_expr: &[F],
    rows: &mut [F],
    width: usize,
    input: &EcMulTraceInput<BLOCKS>,
) -> Result<(), PostflightError> {
    let header_width = ec_mul_header_width();
    let digest_offset = ec_mul_digest_offset(BaseAir::<F>::width(expr));

    let point_bytes = blocks_to_bytes(&input.point_blocks);
    let scalar_bytes = blocks_to_bytes(&input.scalar_blocks);
    let coord_bytes = point_bytes.len() / 2;
    let px = BigUint::from_bytes_le(&point_bytes[..coord_bytes]);
    let py = BigUint::from_bytes_le(&point_bytes[coord_bytes..]);

    // ---- ladder rows -----------------------------------------------------------------
    let mut rx = BigUint::ZERO;
    let mut ry = BigUint::ZERO;
    let mut scalar_acc = BigUint::ZERO;
    let mut is_inf = true;
    let outs = expr.program().output_indices();

    for row_idx in 0..EC_MUL_COMPUTE_ROWS {
        // MSB-first: row 0 consumes bit 255.
        let bit_index = EC_MUL_SCALAR_BITS - 1 - row_idx;
        let bit = (scalar_bytes[bit_index / 8] >> (bit_index % 8)) & 1 == 1;

        let mut flags = vec![false; NUM_STEP_FLAGS];
        if input.is_setup == 0 {
            flags[match (is_inf, bit) {
                (false, false) => FLAG_DBL,
                (false, true) => FLAG_DBL_ADD,
                (true, false) => FLAG_INF_STAY,
                (true, true) => FLAG_INF_TAKE,
            }] = true;
        }

        let row = &mut rows[row_idx * width..(row_idx + 1) * width];
        {
            let header: &mut EcMulHeaderCols<F> = row[..header_width].borrow_mut();
            header.is_compute = F::ONE;
            header.is_digest = F::ZERO;
            header.is_first_compute = if row_idx == 0 { F::ONE } else { F::ZERO };
            let is_setup = input.is_setup != 0;
            header.is_setup = F::from_bool(is_setup);
            header.is_ladder = F::from_bool(!is_setup && row_idx != 0);
            header.row_idx = F::from_usize(row_idx);

            // scalar_acc holds the value before this step; the carries relate it to the next row's
            // value via s' = 2*s + bit.
            let acc_limbs = scalar_acc.to_bytes_le();
            for (i, limb) in header.scalar_acc.iter_mut().enumerate() {
                *limb = F::from_u8(acc_limbs.get(i).copied().unwrap_or(0));
            }
            let mut carry = u16::from(bit && input.is_setup == 0);
            for (i, out) in header.scalar_carry.iter_mut().enumerate() {
                let doubled = u16::from(acc_limbs.get(i).copied().unwrap_or(0)) * 2 + carry;
                carry = doubled >> 8;
                *out = F::from_u16(carry);
            }
            if carry != 0 {
                return Err(PostflightError::new(
                    "EC_MUL scalar accumulator overflowed 256 bits",
                ));
            }
        }

        let inputs = if input.is_setup != 0 {
            setup_row_inputs(expr.program())
        } else {
            vec![rx.clone(), ry.clone(), px.clone(), py.clone()]
        };

        expr.generate_subrow(
            (range_checker, inputs, flags),
            &mut row[header_width..digest_offset],
        );

        // Carry the accumulator to the next row.
        let vars = {
            let sub = &row[header_width..digest_offset];
            let cols = expr.load_vars(sub);
            let read_limbs = |limbs: &[F]| {
                let bytes: Vec<u8> = limbs
                    .iter()
                    .map(|f| u8::try_from(f.as_canonical_u32()).unwrap_or(0))
                    .collect();
                BigUint::from_bytes_le(&bytes)
            };
            (
                read_limbs(&cols.vars[outs[0]]),
                read_limbs(&cols.vars[outs[1]]),
            )
        };
        rx = vars.0;
        ry = vars.1;

        if input.is_setup == 0 {
            scalar_acc = scalar_acc * 2u32 + u32::from(bit);
            is_inf = is_inf && !bit;
        }
    }

    // ---- digest row ------------------------------------------------------------------
    let row = &mut rows[EC_MUL_DIGEST_ROW_IDX * width..(EC_MUL_DIGEST_ROW_IDX + 1) * width];
    {
        let header: &mut EcMulHeaderCols<F> = row[..header_width].borrow_mut();
        header.is_compute = F::ZERO;
        header.is_digest = F::ONE;
        header.is_first_compute = F::ZERO;
        let is_setup = input.is_setup != 0;
        header.is_setup = F::from_bool(is_setup);
        header.is_real_digest = F::from_bool(!is_setup);
        header.row_idx = F::from_usize(EC_MUL_DIGEST_ROW_IDX);
        let acc_limbs = scalar_acc.to_bytes_le();
        for (i, limb) in header.scalar_acc.iter_mut().enumerate() {
            *limb = F::from_u8(acc_limbs.get(i).copied().unwrap_or(0));
        }
    }

    // The expression region is inactive here but cannot be zero, for the same reason padding rows
    // cannot: its ungated carry recurrences do not hold on zeros once `a != 0`.
    row[header_width..digest_offset].copy_from_slice(dummy_expr);

    let digest: &mut EcMulDigestCols<F, NUM_LIMBS, BLOCKS> = row[digest_offset..].borrow_mut();
    digest.from_state.pc = F::from_u32(input.from_pc);
    digest.from_state.timestamp = F::from_u32(input.from_timestamp);
    digest.rs1_ptr = F::from_u32(input.reg_ptrs[0]);
    digest.rs2_ptr = F::from_u32(input.reg_ptrs[1]);
    digest.rd_ptr = F::from_u32(input.reg_ptrs[2]);
    digest.rs1_val = ptr_to_field_u16_limbs(input.reg_vals[0]);
    digest.rs2_val = ptr_to_field_u16_limbs(input.reg_vals[1]);
    digest.rd_val = ptr_to_field_u16_limbs(input.reg_vals[2]);
    for &ptr in &input.reg_vals {
        range_checker.add_count(ptr_bound_from_ptr(ptr, ptr_max_bits), 16);
    }

    let point_bytes = blocks_to_bytes(&input.point_blocks);
    for (i, byte) in point_bytes.iter().enumerate() {
        if i < NUM_LIMBS {
            digest.point_x[i] = F::from_u8(*byte);
        } else {
            digest.point_y[i - NUM_LIMBS] = F::from_u8(*byte);
        }
    }
    for (i, byte) in scalar_bytes.iter().enumerate().take(SCALAR_LIMBS) {
        digest.scalar_data[i] = F::from_u8(*byte);
    }

    let write_bytes = blocks_to_bytes(&input.write_blocks);
    for (i, byte) in write_bytes.iter().enumerate() {
        if i < NUM_LIMBS {
            digest.result_x[i] = F::from_u8(*byte);
        } else {
            digest.result_y[i - NUM_LIMBS] = F::from_u8(*byte);
        }
    }

    // Timestamps run forward in the order the AIR consumes them.
    let mut timestamp = input.from_timestamp;
    let mut next_timestamp = || {
        let current = timestamp;
        timestamp += 1;
        current
    };
    for (prev, aux) in input
        .reg_prev_timestamps
        .iter()
        .zip(digest.rs_read_aux.iter_mut())
    {
        mem_helper.fill(*prev, next_timestamp(), aux.as_mut());
    }
    for (prev, aux) in input
        .point_prev_timestamps
        .iter()
        .zip(digest.point_read_aux.iter_mut())
    {
        mem_helper.fill(*prev, next_timestamp(), aux.as_mut());
    }
    for (prev, aux) in input
        .scalar_prev_timestamps
        .iter()
        .zip(digest.scalar_read_aux.iter_mut())
    {
        mem_helper.fill(*prev, next_timestamp(), aux.as_mut());
    }
    for ((prev, predecessor), aux) in input
        .write_prev_timestamps
        .iter()
        .zip(input.write_predecessors.iter())
        .zip(digest.write_aux.iter_mut())
    {
        let mut bytes = [0u8; MEMORY_BLOCK_BYTES];
        for (dst, &limb) in bytes.chunks_exact_mut(2).zip(predecessor) {
            dst.copy_from_slice(&limb.to_le_bytes());
        }
        aux.set_prev_data(
            openvm_circuit::system::memory::offline_checker::pack_u8_block_bytes(&bytes),
        );
        mem_helper.fill(*prev, next_timestamp(), aux.as_mut());
    }

    Ok(())
}

pub fn generate_ec_mul_trace_from_postflight<
    F: PrimeField32 + Send + Sync,
    const NUM_LIMBS: usize,
    const BLOCKS: usize,
>(
    chip: &EcMulChip<F, NUM_LIMBS, BLOCKS>,
    postflight: &Postflight<'_, F>,
    opcode_base: usize,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let mut selected: Vec<(PostflightStep, bool)> = Vec::new();
    for (local, is_setup) in [
        (WeierstrassOpcode::EC_MUL, false),
        (WeierstrassOpcode::SETUP_EC_MUL, true),
    ] {
        let global = opcode_base
            .checked_add(local as usize)
            .ok_or_else(|| PostflightError::new("EC_MUL opcode overflow"))?;
        selected.extend(
            postflight
                .steps(VmOpcode::from_usize(global))
                .iter()
                .copied()
                .map(|step| (step, is_setup)),
        );
    }
    selected.sort_unstable_by_key(|&(step, _)| postflight.timestamp(step));

    let inputs = selected
        .par_iter()
        .copied()
        .map(|(step, is_setup)| {
            project_step::<F, BLOCKS>(postflight, step, is_setup, chip.ptr_max_bits)
        })
        .collect::<Result<Vec<_>, PostflightError>>()?;

    build_ec_mul_trace::<F, NUM_LIMBS, BLOCKS>(chip, &inputs)
}

/// Fills the trace from already-replayed instruction data.
///
/// Split from the postflight walk above so both prover backends share one row layout: the CPU
/// prover projects from a host [`Postflight`], while the GPU prover gathers the same fields from
/// the device transcript. Neither can drift from the other's row encoding.
pub(crate) fn build_ec_mul_trace<
    F: PrimeField32 + Send + Sync,
    const NUM_LIMBS: usize,
    const BLOCKS: usize,
>(
    chip: &EcMulChip<F, NUM_LIMBS, BLOCKS>,
    inputs: &[EcMulTraceInput<BLOCKS>],
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let width = ec_mul_width::<NUM_LIMBS, BLOCKS>(BaseAir::<F>::width(&chip.expr));
    let used_rows = inputs.len() * EC_MUL_TOTAL_ROWS;
    let height = if used_rows == 0 {
        0
    } else {
        used_rows
            .checked_next_power_of_two()
            .ok_or_else(|| PostflightError::new("EC_MUL trace height overflow"))?
    };
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    // An inactive expression region cannot be zero: the curve's `a` coefficient is folded in as a
    // constant, so on an all-zero row the lambda constraint evaluates to `-a` and the ungated carry
    // recurrences are unsatisfiable whenever `a != 0`. Build one consistent witness for zero inputs
    // with `is_valid` cleared, as `fill_dummy_core_row` does for the single-row chips, and reuse it
    // for every digest and padding row. Its range-check counts are discarded, since the AIR emits
    // no range checks when `is_valid` is zero.
    let expr_width = BaseAir::<F>::width(&chip.expr);
    let dummy_expr = {
        let discard = VariableRangeCheckerChip::new(chip.range_checker.bus());
        let mut sub = F::zero_vec(expr_width);
        chip.expr.generate_subrow(
            (
                &discard,
                vec![BigUint::ZERO; chip.expr.program().num_inputs()],
                vec![false; NUM_STEP_FLAGS],
            ),
            &mut sub,
        );
        sub[0] = F::ZERO;
        sub
    };

    // Counts accumulate into a private chip and are merged at the end so the per-instruction fills
    // can run in parallel, as the single-row chips do.
    let counter = Arc::new(VariableRangeCheckerChip::new(chip.range_checker.bus()));
    let helper_owner =
        SharedMemoryHelper::new(counter.clone(), chip.mem_helper.timestamp_max_bits());
    let mem_helper = helper_owner.as_borrowed();

    trace.values[..used_rows * width]
        .par_chunks_exact_mut(EC_MUL_TOTAL_ROWS * width)
        .zip(inputs.par_iter())
        .try_for_each(|(rows, input)| {
            fill_instruction::<F, NUM_LIMBS, BLOCKS>(
                &chip.expr,
                counter.as_ref(),
                &mem_helper,
                chip.ptr_max_bits,
                &dummy_expr,
                rows,
                width,
                input,
            )
        })?;

    if used_rows < height {
        let mut dummy = F::zero_vec(width);
        dummy[ec_mul_header_width()..ec_mul_digest_offset(expr_width)].copy_from_slice(&dummy_expr);
        trace.values[used_rows * width..]
            .par_chunks_exact_mut(width)
            .for_each(|row| row.copy_from_slice(&dummy));
    }

    if chip.range_checker.count.len() != counter.count.len() {
        return Err(PostflightError::new("EC_MUL range-count shape mismatch"));
    }
    for (dst, src) in chip.range_checker.count.iter().zip(&counter.count) {
        dst.fetch_add(src.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    Ok(trace)
}
