use openvm_circuit::{
    arch::{Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::LOADW;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{Rv64LoadSignExtendWordChip, LOAD_SIGN_EXTEND_WORD_OVERLAP_CELLS};
use crate::{
    adapters::Rv64LoadMultiByteAdapterCols, load_sign_extend::core::LoadSignExtendCoreCols,
};

/// Generates the signed word-load trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64LoadSignExtendWordChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(LOADW.global_opcode());
    let width = Rv64LoadMultiByteAdapterCols::<F>::width()
        + LoadSignExtendCoreCols::<F, LOAD_SIGN_EXTEND_WORD_OVERLAP_CELLS>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    for (row_index, &step) in steps.iter().enumerate() {
        let row = &mut trace.values[row_index * width..(row_index + 1) * width];
        chip.inner
            .replay(postflight, step, &chip.mem_helper.as_borrowed(), row)?;
    }

    Ok(trace)
}
