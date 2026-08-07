use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{riscv::REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::DivRemOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_divrem, DivRemChip, DivRemCoreCols};
use crate::adapters::{MultAdapterCols, MultAdapterFiller, ReplayComputation, BYTE_BITS};

/// Generates the RV64 DIV/DIVU/REM/REMU trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &DivRemChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [
        DivRemOpcode::DIV,
        DivRemOpcode::DIVU,
        DivRemOpcode::REM,
        DivRemOpcode::REMU,
    ];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = MultAdapterCols::<F>::width();
    let width = adapter_width + DivRemCoreCols::<F, REGISTER_NUM_LIMBS, BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for opcode in opcodes {
        let steps = postflight.steps(opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let core_row: &mut DivRemCoreCols<F, REGISTER_NUM_LIMBS, BYTE_BITS> =
                core_row.borrow_mut();
            let is_signed = opcode == DivRemOpcode::DIV || opcode == DivRemOpcode::REM;
            let is_div = opcode == DivRemOpcode::DIV || opcode == DivRemOpcode::DIVU;
            let replay = MultAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |[b, c]| {
                    let computed = run_divrem::<REGISTER_NUM_LIMBS, BYTE_BITS>(
                        is_signed,
                        &b.map(u32::from),
                        &c.map(u32::from),
                    );
                    let output =
                        (if is_div { computed.0 } else { computed.1 }).map(|value| value as u8);
                    ReplayComputation {
                        output,
                        metadata: computed,
                    }
                },
            )?;
            let [b, c] = replay.inputs;
            chip.inner
                .fill_core_row_with_result(opcode, b, c, replay.metadata, core_row);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
