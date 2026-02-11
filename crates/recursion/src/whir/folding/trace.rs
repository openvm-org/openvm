use core::{borrow::BorrowMut, convert::TryInto};

use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{EF, F};

use super::WhirFoldingCols;
use crate::{
    tracegen::{RowMajorChip, StandardTracegenCtx},
    whir::total_num_queries,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FoldRecord {
    pub whir_round: u32,
    pub query_idx: u32,
    pub coset_idx: u32,
    pub height: u32,
    pub coset_size: u32,
    pub coset_shift: F,
    pub twiddle: F,
    pub z_final: F,
    pub value: EF,
    pub left_value: EF,
    pub right_value: EF,
    pub y_final: EF,
    pub alpha: EF,
}

impl FoldRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        whir_round: usize,
        query_idx: usize,
        twiddle: F,
        coset_shift: F,
        coset_size: usize,
        coset_idx: usize,
        height: usize,
        left_value: EF,
        right_value: EF,
        value: EF,
        alpha: EF,
    ) -> Self {
        debug_assert!(height > 0, "folding record height must be > 0");
        Self {
            whir_round: whir_round.try_into().unwrap(),
            query_idx: query_idx.try_into().unwrap(),
            coset_idx: coset_idx.try_into().unwrap(),
            height: height.try_into().unwrap(),
            coset_size: coset_size.try_into().unwrap(),
            coset_shift,
            twiddle,
            value,
            left_value,
            right_value,
            z_final: F::ZERO,
            y_final: EF::ZERO,
            alpha,
        }
    }

    pub fn set_final_values(&mut self, z_final: F, y_final: EF) {
        self.z_final = z_final;
        self.y_final = y_final;
    }
}

pub(crate) struct FoldingTraceGenerator;

impl RowMajorChip<F> for FoldingTraceGenerator {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let mvk = ctx.vk;
        let preflights = ctx.preflights;
        let params = &mvk.inner.params;

        let num_queries_per_round: Vec<usize> =
            params.whir.rounds.iter().map(|r| r.num_queries).collect();
        let k_whir = params.k_whir();
        let internal_nodes = (1 << k_whir) - 1;

        let num_rows_per_proof = total_num_queries(&num_queries_per_round) * internal_nodes;
        let num_valid_rows = num_rows_per_proof * preflights.len();
        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = WhirFoldingCols::<F>::width();

        let mut trace = vec![F::ZERO; height * width];

        for (row_idx, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
            let proof_idx = row_idx / num_rows_per_proof;
            let i = row_idx % num_rows_per_proof;

            let preflight = &preflights[proof_idx];

            let record = preflight.whir.fold_records[i];
            let height = record.height as usize;

            let cols: &mut WhirFoldingCols<F> = row.borrow_mut();
            cols.is_valid = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_root = F::from_bool(record.coset_size == 1);
            cols.alpha
                .copy_from_slice(record.alpha.as_basis_coefficients_slice());
            cols.height = F::from_usize(height);
            cols.whir_round = F::from_u32(record.whir_round);
            cols.query_idx = F::from_u32(record.query_idx);
            cols.coset_idx = F::from_u32(record.coset_idx);
            cols.left_value
                .copy_from_slice(record.left_value.as_basis_coefficients_slice());
            cols.right_value
                .copy_from_slice(record.right_value.as_basis_coefficients_slice());
            cols.value
                .copy_from_slice(record.value.as_basis_coefficients_slice());
            cols.twiddle = record.twiddle;
            cols.coset_shift = record.coset_shift;
            cols.coset_size = F::from_u32(record.coset_size);
            cols.z_final = record.z_final;
            cols.y_final
                .copy_from_slice(record.y_final.as_basis_coefficients_slice());
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}
