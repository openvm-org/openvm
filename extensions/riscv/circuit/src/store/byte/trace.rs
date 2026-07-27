use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::STOREB;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{Rv64StoreByteChip, StoreByteCoreCols};
use crate::adapters::Rv64StoreByteAdapterCols;

/// Generates the byte-store trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64StoreByteChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(STOREB.global_opcode());
    let adapter_width = Rv64StoreByteAdapterCols::<F>::width();
    let width = adapter_width + StoreByteCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    for (row_index, &step) in steps.iter().enumerate() {
        let row = &mut trace.values[row_index * width..(row_index + 1) * width];
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        chip.inner.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            core_row.borrow_mut(),
        )?;
    }
    for row_index in steps.len()..height {
        let row = &mut trace.values[row_index * width..(row_index + 1) * width];
        let adapter_row: &mut Rv64StoreByteAdapterCols<F> = row[..adapter_width].borrow_mut();
        adapter_row.mem_as = F::from_u32(2);
    }

    Ok(trace)
}
