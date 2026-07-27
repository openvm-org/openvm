use openvm_circuit::arch::{Postflight, PostflightError};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{Rv64LoadHalfwordChip, LOAD_HALFWORD_OVERLAP_CELLS};
use crate::{adapters::HALFWORD_ACCESS_WIDTH, load::core};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64LoadHalfwordChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    core::generate_trace_from_postflight::<F, HALFWORD_ACCESS_WIDTH, LOAD_HALFWORD_OVERLAP_CELLS>(
        chip, postflight,
    )
}
