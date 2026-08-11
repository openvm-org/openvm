use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::MulOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{fill_core_row_with_result, run_mul, MultiplicationChip, MultiplicationCoreCols};
use crate::adapters::{
    MultAdapterCols, MultAdapterFiller, ReplayComputation, BYTE_BITS, REGISTER_NUM_LIMBS,
};

/// Generates the RV64 MUL trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &MultiplicationChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = MulOpcode::MUL.global_opcode();
    let rows_used = postflight.steps(opcode).len();
    let adapter_width = MultAdapterCols::<F>::width();
    let width = adapter_width + MultiplicationCoreCols::<F, REGISTER_NUM_LIMBS, BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, postflight.steps(opcode), |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let replay = MultAdapterFiller::replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |[rs1, rs2]| {
                let (output, computed_carry) = run_mul::<REGISTER_NUM_LIMBS, BYTE_BITS>(&rs1, &rs2);
                ReplayComputation {
                    output,
                    metadata: computed_carry,
                }
            },
        )?;
        let [rs1, rs2] = replay.inputs;
        let core_row: &mut MultiplicationCoreCols<F, REGISTER_NUM_LIMBS, BYTE_BITS> =
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
