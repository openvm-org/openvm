use openvm_circuit::arch::{Postflight, PostflightError};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{StoreHalfwordChip, STORE_HALFWORD_VALUE_CELLS};
use crate::{adapters::HALFWORD_ACCESS_WIDTH, store::core};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &StoreHalfwordChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    core::generate_trace_from_postflight::<F, HALFWORD_ACCESS_WIDTH, STORE_HALFWORD_VALUE_CELLS>(
        chip, postflight,
    )
}
