use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LoadStoreOpcode::LOADB;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{LoadSignExtendByteChip, LoadSignExtendByteCoreCols};
use crate::{
    adapters::{
        u16_cell_byte, LoadByteAdapterCols, BYTE_BITS, BYTE_SHIFT_SELECTOR_WIDTH, BYTE_SIGN_BIT,
    },
    load_sign_extend::common::load_sign_extend_write_data,
};

/// Generates the signed byte-load trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &LoadSignExtendByteChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(LOADB.global_opcode());
    let adapter_width = LoadByteAdapterCols::<F>::width();
    let width = adapter_width + LoadSignExtendByteCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let (read_data, shift, _) = chip.inner.adapter.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |read_data, shift| {
                load_sign_extend_write_data(LOADB, [read_data, [0; BLOCK_FE_WIDTH]], shift)
            },
        )?;

        let core_row: &mut LoadSignExtendByteCoreCols<F> = core_row.borrow_mut();
        let read_cell = read_data[shift / 2];
        let read_cell_bytes = [u16_cell_byte(read_cell, 0), u16_cell_byte(read_cell, 1)];
        chip.inner
            .bitwise_lookup_chip
            .request_range(read_cell_bytes[0] as u32, read_cell_bytes[1] as u32);
        core_row.read_cell_lo_byte = F::from_u16(read_cell_bytes[0]);

        let byte = read_cell_bytes[shift % 2];
        let sign_bit = byte & BYTE_SIGN_BIT;
        chip.inner
            .range_checker_chip
            .add_count((byte - sign_bit) as u32, BYTE_BITS - 1);
        core_row.data_most_sig_bit = F::from_bool(sign_bit != 0);
        core_row.read_data = read_data.map(F::from_u16);
        let selector: &[u32; BYTE_SHIFT_SELECTOR_WIDTH] =
            chip.inner.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*selector).map(F::from_u32);
        Ok(())
    })?;

    Ok(trace)
}
