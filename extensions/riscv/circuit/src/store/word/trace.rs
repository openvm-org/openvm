use openvm_circuit::arch::{Postflight, PostflightError};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{StoreWordChip, STORE_WORD_VALUE_CELLS};
use crate::{adapters::WORD_ACCESS_WIDTH, store::core};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &StoreWordChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    core::generate_trace_from_postflight::<F, WORD_ACCESS_WIDTH, STORE_WORD_VALUE_CELLS>(
        chip, postflight,
    )
}
