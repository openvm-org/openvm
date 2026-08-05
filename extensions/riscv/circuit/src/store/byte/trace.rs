use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LoadStoreOpcode::STOREB;
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};

use super::{StoreByteChip, StoreByteCoreCols};
use crate::adapters::StoreByteAdapterCols;

/// Generates the byte-store trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &StoreByteChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(STOREB.global_opcode());
    let adapter_width = StoreByteAdapterCols::<F>::width();
    let width = adapter_width + StoreByteCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        chip.inner.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            core_row.borrow_mut(),
        )?;
        Ok(())
    })?;
    trace.values[steps.len() * width..]
        .par_chunks_exact_mut(width)
        .for_each(fill_padding_row);
    Ok(trace)
}

pub(crate) fn fill_padding_row<F: PrimeField32>(row: &mut [F]) {
    let adapter_width = StoreByteAdapterCols::<F>::width();
    let adapter_row: &mut StoreByteAdapterCols<F> = row[..adapter_width].borrow_mut();
    adapter_row.mem_as = F::from_u32(2);
}
