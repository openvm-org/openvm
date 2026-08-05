use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{RevealChip, RevealCols};
use crate::adapters::{byte_ptr_to_u16_ptr_value, checked_register_pointer};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &RevealChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(RevealOpcode::REVEAL.global_opcode());
    let width = RevealCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let instruction = postflight.instruction(step);
        if instruction.opcode != RevealOpcode::REVEAL.global_opcode()
            || !instruction.b.is_zero()
            || !instruction.c.is_zero()
            || !instruction.d.is_zero()
            || !instruction.e.is_zero()
            || !instruction.f.is_zero()
            || !instruction.g.is_zero()
        {
            return Err(PostflightError::new(
                "REVEAL instruction has invalid fixed operands",
            ));
        }
        let src_ptr = u32::from(checked_register_pointer(instruction.a.as_canonical_u32())?);
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let mut replay = postflight.replay(step);
        let src_read = replay.read_u16(REGISTER_AS, byte_ptr_to_u16_ptr_value(src_ptr))?;
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        let cols: &mut RevealCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.from_state.pc = F::from_u32(from_pc);
        cols.from_state.timestamp = F::from_u32(from_timestamp);
        cols.src_ptr = F::from_u32(src_ptr);
        cols.src_data = src_read.value.map(F::from_u16);
        mem_helper.fill(
            src_read.previous_timestamp,
            src_read.timestamp,
            cols.src_aux.as_mut(),
        );
        Ok(())
    })?;

    // Fill segment-local ordering and timestamp-gap witnesses.
    let low_bits = chip
        .inner
        .timestamp_max_bits
        .min(chip.inner.range_checker_chip.range_max_bits());
    let high_bits = chip.inner.timestamp_max_bits - low_bits;
    let low_mask = (1u32 << low_bits) - 1;
    for (ordinal, row) in trace
        .values
        .chunks_exact_mut(width)
        .take(steps.len())
        .enumerate()
    {
        let cols: &mut RevealCols<F> = row.borrow_mut();
        cols.ordinal = F::from_usize(ordinal);
        let Some(next_step) = steps.get(ordinal + 1) else {
            continue;
        };
        cols.has_next = F::ONE;
        let timestamp_delta = postflight
            .timestamp(*next_step)
            .checked_sub(postflight.timestamp(steps[ordinal]))
            .and_then(|delta| delta.checked_sub(1))
            .ok_or_else(|| PostflightError::new("REVEAL timestamps are not strictly increasing"))?;
        let low = timestamp_delta & low_mask;
        let high = timestamp_delta >> low_bits;
        cols.timestamp_delta_low = F::from_u32(low);
        chip.inner.range_checker_chip.add_count(low, low_bits);
        chip.inner.range_checker_chip.add_count(high, high_bits);
    }

    Ok(trace)
}
