use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::MulHOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{
    core::{fill_core_row_with_result, run_mulh},
    MulHChip, MulHCoreCols,
};
use crate::adapters::{
    MultAdapterCols, MultAdapterFiller, ReplayComputation, BYTE_BITS, REGISTER_NUM_LIMBS,
};

/// Generates the RV64 multiply-high trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &MulHChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [MulHOpcode::MULH, MulHOpcode::MULHSU, MulHOpcode::MULHU];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = MultAdapterCols::<F>::width();
    let width = adapter_width + MulHCoreCols::<F, REGISTER_NUM_LIMBS, BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for opcode in opcodes {
        let steps = postflight.steps(opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let replay = MultAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |[b, c]| {
                    let computed = run_mulh::<REGISTER_NUM_LIMBS, BYTE_BITS>(
                        opcode,
                        &b.map(u32::from),
                        &c.map(u32::from),
                    );
                    let output = computed.0.map(|limb| limb as u8);
                    ReplayComputation {
                        output,
                        metadata: computed,
                    }
                },
            )?;
            let [b, c] = replay.inputs;
            let core_row: &mut MulHCoreCols<F, REGISTER_NUM_LIMBS, BYTE_BITS> =
                core_row.borrow_mut();
            fill_core_row_with_result(
                &chip.inner.range_tuple_chip,
                &chip.inner.bitwise_lookup_chip,
                core_row,
                opcode,
                b,
                c,
                replay.metadata,
            );
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
