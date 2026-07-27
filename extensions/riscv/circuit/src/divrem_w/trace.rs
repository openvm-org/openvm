use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{riscv::RV64_WORD_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::{DivRemOpcode, DivRemWOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::Rv64DivRemWChip;
use crate::{
    adapters::{Rv64MultWAdapterCols, RV64_BYTE_BITS},
    divrem::{run_divrem, DivRemCoreCols},
};

/// Generates the RV64 DIVW/DIVUW/REMW/REMUW trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64DivRemWChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [
        DivRemWOpcode::DIVW,
        DivRemWOpcode::DIVUW,
        DivRemWOpcode::REMW,
        DivRemWOpcode::REMUW,
    ];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = Rv64MultWAdapterCols::<F>::width();
    let width = adapter_width + DivRemCoreCols::<F, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for opcode in opcodes {
        for &step in postflight.steps(opcode.global_opcode()) {
            let row = &mut trace.values[row_index * width..(row_index + 1) * width];
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let core_row: &mut DivRemCoreCols<F, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS> =
                core_row.borrow_mut();
            let is_signed = opcode == DivRemWOpcode::DIVW || opcode == DivRemWOpcode::REMW;
            let is_div = opcode == DivRemWOpcode::DIVW || opcode == DivRemWOpcode::DIVUW;
            let core_opcode = match opcode {
                DivRemWOpcode::DIVW => DivRemOpcode::DIV,
                DivRemWOpcode::DIVUW => DivRemOpcode::DIVU,
                DivRemWOpcode::REMW => DivRemOpcode::REM,
                DivRemWOpcode::REMUW => DivRemOpcode::REMU,
            };
            let mut result = None;
            let ([b, c], _) = chip.inner.adapter.replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |[b, c]| {
                    let computed = run_divrem::<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>(
                        is_signed,
                        &b.map(u32::from),
                        &c.map(u32::from),
                    );
                    let output =
                        (if is_div { computed.0 } else { computed.1 }).map(|value| value as u8);
                    result = Some(computed);
                    output
                },
            )?;
            chip.inner.fill_core_row_with_result(
                core_opcode,
                b,
                c,
                result.expect("word divrem replay closure always runs"),
                core_row,
            );
            row_index += 1;
        }
    }

    Ok(trace)
}
