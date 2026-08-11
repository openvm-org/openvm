use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LoadStoreOpcode::LOADBU;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{LoadByteChip, LoadByteCoreCols};
use crate::{
    adapters::{u16_cell_byte, LoadByteAdapterCols, BYTE_SHIFT_SELECTOR_WIDTH},
    load::common::load_byte_write_data,
};

/// Generates the unsigned byte-load trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &LoadByteChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(LOADBU.global_opcode());
    let adapter_width = LoadByteAdapterCols::<F>::width();
    let width = adapter_width + LoadByteCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let (read_data, shift, _) = chip.inner.adapter.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            load_byte_write_data,
        )?;

        let core_row: &mut LoadByteCoreCols<F> = core_row.borrow_mut();
        let read_cell = read_data[shift / 2];
        let read_cell_bytes = [u16_cell_byte(read_cell, 0), u16_cell_byte(read_cell, 1)];
        chip.inner
            .bitwise_lookup_chip
            .request_range(read_cell_bytes[0] as u32, read_cell_bytes[1] as u32);
        core_row.read_cell_lo_byte = F::from_u16(read_cell_bytes[0]);
        core_row.read_data = read_data.map(F::from_u16);
        let selector: &[u32; BYTE_SHIFT_SELECTOR_WIDTH] =
            chip.inner.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*selector).map(F::from_u32);
        Ok(())
    })?;

    Ok(trace)
}
