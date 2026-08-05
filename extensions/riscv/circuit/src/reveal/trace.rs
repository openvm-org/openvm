use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{RevealReplay, RevealAdapterCols, RevealChip, RevealCoreCols};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &RevealChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(RevealOpcode::REVEAL.global_opcode());
    let adapter_width = RevealAdapterCols::<F>::width();
    let width = adapter_width + RevealCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let RevealReplay {
            src_data,
            prev_data,
            shift,
        } = chip.inner.adapter.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
        )?;
        chip.inner
            .fill_core_row(shift, src_data, prev_data, core_row.borrow_mut());
        Ok(())
    })?;
    Ok(trace)
}
