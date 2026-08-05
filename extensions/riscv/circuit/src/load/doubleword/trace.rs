use openvm_circuit::arch::{Postflight, PostflightError};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{LoadDoublewordChip, LOAD_DOUBLEWORD_OVERLAP_CELLS};
use crate::{adapters::DOUBLEWORD_ACCESS_WIDTH, load::core};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &LoadDoublewordChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    core::generate_trace_from_postflight::<F, DOUBLEWORD_ACCESS_WIDTH, LOAD_DOUBLEWORD_OVERLAP_CELLS>(
        chip, postflight,
    )
}
