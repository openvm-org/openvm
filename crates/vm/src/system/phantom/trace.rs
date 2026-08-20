use std::borrow::BorrowMut;

use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, SystemOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::PhantomCols;
use crate::{
    arch::{Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};

/// Generates the phantom trace directly from immutable preflight history.
///
/// Phantom host callbacks have already run during serial preflight. Replay only
/// validates the logged program transition and emits its execution-bus row.
pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(SystemOpcode::PHANTOM.global_opcode());
    let width = PhantomCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    for (row_index, &step) in steps.iter().enumerate() {
        let instruction = postflight.instruction(step);
        let Some(operands) = instruction.checked_phantom_operands() else {
            return Err(PostflightError::new(format!(
                "phantom instruction at PC {:#x} has invalid operands",
                postflight.pc(step)
            )));
        };

        let pc = postflight.pc(step);
        let timestamp = postflight.timestamp(step);
        let mut replay = postflight.replay(step);
        replay.advance_timestamp(1)?;
        replay.finish(pc.wrapping_add(DEFAULT_PC_STEP))?;

        let row: &mut PhantomCols<F> =
            trace.values[row_index * width..(row_index + 1) * width].borrow_mut();
        row.pc = F::from_u32(pc);
        row.operands = operands.map(F::from_u32);
        row.timestamp = F::from_u32(timestamp);
        row.is_valid = F::ONE;
    }

    Ok(trace)
}
