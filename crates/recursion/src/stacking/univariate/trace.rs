use std::borrow::BorrowMut;

use itertools::{izip, Itertools};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    stacking::univariate::air::UnivariateRoundCols,
    tracegen::{RowMajorChip, StandardTracegenCtx},
};

pub struct UnivariateRoundTraceGenerator;

impl RowMajorChip<F> for UnivariateRoundTraceGenerator {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let vk = ctx.vk;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = UnivariateRoundCols::<usize>::width();

        let traces = proofs
            .par_iter()
            .zip(preflights.par_iter())
            .enumerate()
            .map(|(proof_idx, (proof, preflight))| {
                let coeffs = &proof.stacking_proof.univariate_round_coeffs;
                let num_rows = coeffs.len();
                let proof_idx_value = F::from_usize(proof_idx);

                let mut trace = vec![F::ZERO; num_rows * width];

                let u_0 = preflight.stacking.sumcheck_rnd[0];
                let u_0_pows = u_0.powers().take(num_rows).collect_vec();

                let initial_tidx = preflight.stacking.intermediate_tidx[0];

                let d_card = 1usize << vk.inner.params.l_skip;
                let mut s_0_sum_over_d = coeffs[0] * F::from_usize(d_card);
                let mut poly_rand_eval = EF::ZERO;

                for (i, (&coeff, chunk, &u_0_pow)) in
                    izip!(coeffs.iter(), trace.chunks_mut(width), u_0_pows.iter()).enumerate()
                {
                    let cols: &mut UnivariateRoundCols<F> = chunk.borrow_mut();
                    cols.proof_idx = proof_idx_value;
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(i == 0);
                    cols.is_last = F::from_bool(i + 1 == num_rows);

                    cols.tidx = F::from_usize(initial_tidx + (D_EF * i));
                    cols.u_0.copy_from_slice(u_0.as_basis_coefficients_slice());
                    cols.u_0_pow
                        .copy_from_slice(u_0_pow.as_basis_coefficients_slice());

                    cols.coeff
                        .copy_from_slice(coeff.as_basis_coefficients_slice());

                    cols.coeff_idx = F::from_usize(i);
                    if i == d_card {
                        s_0_sum_over_d += coeff * F::from_usize(d_card);
                        cols.coeff_is_d = F::ONE;
                    }
                    cols.s_0_sum_over_d
                        .copy_from_slice(s_0_sum_over_d.as_basis_coefficients_slice());

                    poly_rand_eval += coeff * u_0_pow;
                    cols.poly_rand_eval
                        .copy_from_slice(poly_rand_eval.as_basis_coefficients_slice());
                }

                (trace, num_rows)
            })
            .collect::<Vec<_>>();

        let num_valid_rows = traces.iter().map(|(_trace, num_rows)| *num_rows).sum();
        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };

        let mut combined_trace = Vec::with_capacity(height * width);
        for (trace, _num_rows) in traces {
            combined_trace.extend(trace);
        }

        let padding_proof_idx = F::from_usize(proofs.len());
        combined_trace.resize(height * width, F::ZERO);
        let mut chunks = combined_trace[num_valid_rows * width..]
            .chunks_mut(width)
            .peekable();

        while let Some(chunk) = chunks.next() {
            let cols: &mut UnivariateRoundCols<F> = chunk.borrow_mut();
            cols.proof_idx = padding_proof_idx;
            if chunks.peek().is_none() {
                cols.is_last = F::ONE;
            }
        }

        Some(RowMajorMatrix::new(combined_trace, width))
    }
}
