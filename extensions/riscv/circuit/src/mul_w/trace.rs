use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::MulWOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::MulWChip;
use crate::{
    adapters::{
        MultWAdapterCols, MultWAdapterFiller, ReplayComputation, BYTE_BITS, WORD_NUM_LIMBS,
    },
    mul::{fill_core_row_with_result, run_mul, MultiplicationCoreCols},
};

/// Generates the RV64 MULW trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &MulWChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = MulWOpcode::MULW.global_opcode();
    let rows_used = postflight.steps(opcode).len();
    let adapter_width = MultWAdapterCols::<F>::width();
    let width = adapter_width + MultiplicationCoreCols::<F, WORD_NUM_LIMBS, BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let adapter = MultWAdapterFiller::new(chip.inner.bitwise_lookup_chip.clone());

    fill_trace_rows(&mut trace, 0, postflight.steps(opcode), |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let replay = adapter.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |[rs1, rs2]| {
                let (output, computed_carry) = run_mul::<WORD_NUM_LIMBS, BYTE_BITS>(&rs1, &rs2);
                ReplayComputation {
                    output,
                    metadata: computed_carry,
                }
            },
        )?;
        let [rs1, rs2] = replay.inputs;
        let core_row: &mut MultiplicationCoreCols<F, WORD_NUM_LIMBS, BYTE_BITS> =
            core_row.borrow_mut();
        fill_core_row_with_result(
            &chip.inner.range_tuple_chip,
            &chip.inner.bitwise_lookup_chip,
            core_row,
            rs1,
            rs2,
            replay.output,
            replay.metadata,
        );
        Ok(())
    })?;

    Ok(trace)
}
