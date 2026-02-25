use std::borrow::BorrowMut;

use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{proof_shape::pvs::air::PublicValuesCols, system::Preflight, tracegen::RowMajorChip};

pub struct PublicValuesTraceGenerator;

impl RowMajorChip<F> for PublicValuesTraceGenerator {
    type Ctx<'a> = (&'a [Proof<BabyBearPoseidon2Config>], &'a [Preflight]);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (proofs, preflights) = ctx;
        let num_valid_rows = proofs
            .iter()
            .map(|proof| {
                proof
                    .public_values
                    .iter()
                    .fold(0usize, |acc, per_air| acc + per_air.len())
            })
            .sum();
        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = PublicValuesCols::<u8>::width();

        debug_assert_eq!(proofs.len(), preflights.len());

        let mut trace = vec![F::ZERO; height * width];
        let mut chunks = trace.chunks_exact_mut(width);

        for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights.iter()).enumerate() {
            let mut row_idx = 0usize;

            for ((air_idx, pvs), &starting_tidx) in proof
                .public_values
                .iter()
                .enumerate()
                .filter(|(_, per_air)| !per_air.is_empty())
                .zip(&preflight.proof_shape.pvs_tidx)
            {
                let mut tidx = starting_tidx;

                for (pv_idx, pv) in pvs.iter().enumerate() {
                    let chunk = chunks.next().unwrap();
                    let cols: &mut PublicValuesCols<F> = chunk.borrow_mut();

                    cols.is_valid = F::ONE;

                    cols.proof_idx = F::from_usize(proof_idx);
                    cols.air_idx = F::from_usize(air_idx);
                    cols.pv_idx = F::from_usize(pv_idx);

                    cols.is_first_in_air = F::from_bool(pv_idx == 0);
                    cols.is_first_in_proof = F::from_bool(row_idx == 0);

                    cols.tidx = F::from_usize(tidx);
                    cols.value = *pv;

                    row_idx += 1;
                    tidx += 1;
                }
            }
        }

        for chunk in chunks {
            let cols: &mut PublicValuesCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_usize(proofs.len());
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}
