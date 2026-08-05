use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, VmChipWrapper},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};

use super::{RevealAdapterCols, RevealFiller};
use crate::store::{
    common::doubleword_rmw_write_data,
    core::{fill_padding_row, StoreCoreCols},
    STORE_DOUBLEWORD_VALUE_CELLS,
};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &VmChipWrapper<F, RevealFiller>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(RevealOpcode::REVEAL.global_opcode());
    let adapter_width = RevealAdapterCols::<F>::width();
    let width = adapter_width + StoreCoreCols::<F, STORE_DOUBLEWORD_VALUE_CELLS>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let (read_data, prev_data, shift) = chip.inner.inner.adapter.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            doubleword_rmw_write_data,
        )?;
        chip.inner
            .inner
            .fill_core_row(shift, read_data, prev_data, core_row.borrow_mut());
        Ok(())
    })?;
    trace.values[steps.len() * width..]
        .par_chunks_exact_mut(width)
        .for_each(fill_padding_row);
    Ok(trace)
}
