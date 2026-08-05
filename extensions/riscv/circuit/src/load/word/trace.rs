use openvm_circuit::arch::{Postflight, PostflightError};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{LoadWordChip, LOAD_WORD_OVERLAP_CELLS};
use crate::{adapters::WORD_ACCESS_WIDTH, load::core};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &LoadWordChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    core::generate_trace_from_postflight::<F, WORD_ACCESS_WIDTH, LOAD_WORD_OVERLAP_CELLS>(
        chip, postflight,
    )
}
