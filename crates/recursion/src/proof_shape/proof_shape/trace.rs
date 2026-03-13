use std::{array::from_fn, borrow::BorrowMut, sync::Arc};

use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    primitives::{pow::PowerCheckerCpuTraceGenerator, range::RangeCheckerCpuTraceGenerator},
    proof_shape::proof_shape::air::{
        borrow_var_cols_mut, decompose_f, decompose_usize, ProofShapeCols, ProofShapeVarColsMut,
    },
    system::{Preflight, POW_CHECKER_HEIGHT},
    tracegen::RowMajorChip,
    utils::interaction_length,
};

pub(crate) fn compute_air_shape_lookup_counts(
    child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
) -> Vec<usize> {
    child_vk
        .inner
        .per_air
        .iter()
        .map(|avk| {
            let dag = &avk.symbolic_constraints;
            dag.constraints.nodes.len()
                + avk.unused_variables.len()
                + dag
                    .interactions
                    .iter()
                    .map(interaction_length)
                    .sum::<usize>()
        })
        .collect::<Vec<_>>()
}

#[derive(derive_new::new)]
pub(in crate::proof_shape) struct ProofShapeChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    idx_encoder: Arc<Encoder>,
    min_cached_idx: usize,
    max_cached: usize,
    range_checker: Arc<RangeCheckerCpuTraceGenerator<LIMB_BITS>>,
    pow_checker: Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> RowMajorChip<F>
    for ProofShapeChip<NUM_LIMBS, LIMB_BITS>
{
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a [Proof<BabyBearPoseidon2Config>],
        &'a [Preflight],
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (child_vk, proofs, preflights) = ctx;
        let num_valid_rows = proofs.len() * (child_vk.inner.per_air.len() + 1);
        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
        let idx_encoder = &self.idx_encoder;
        let min_cached_idx = self.min_cached_idx;
        let max_cached = self.max_cached;
        let range_checker = &self.range_checker;
        let pow_checker = &self.pow_checker;
        let num_airs = child_vk.inner.per_air.len();
        let cols_width = ProofShapeCols::<usize, NUM_LIMBS>::width();
        let total_width = self.idx_encoder.width() + cols_width + self.max_cached * DIGEST_SIZE;
        let l_skip = child_vk.inner.params.l_skip;

        debug_assert_eq!(proofs.len(), preflights.len());

        let mut trace = vec![F::ZERO; height * total_width];
        let mut chunks = trace.chunks_exact_mut(total_width);

        for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights.iter()).enumerate() {
            let mut sorted_idx = 0usize;
            let mut total_interactions = 0usize;
            let mut cidx = 1usize;
            let mut num_present = 0usize;
            let bc_air_shape_lookups = compute_air_shape_lookup_counts(child_vk);

            // Present AIRs
            for (idx, vdata) in &preflight.proof_shape.sorted_trace_vdata {
                let chunk = chunks.next().unwrap();
                let cols: &mut ProofShapeCols<F, NUM_LIMBS> = chunk[..cols_width].borrow_mut();
                let log_height = vdata.log_height;
                let height = 1 << log_height;
                let n = log_height as isize - l_skip as isize;
                num_present += 1;

                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(sorted_idx == 0);

                cols.sorted_idx = F::from_usize(sorted_idx);
                cols.log_height = F::from_usize(log_height);
                cols.n_sign_bit = F::from_bool(n.is_negative());
                sorted_idx += 1;

                cols.starting_tidx = F::from_usize(preflight.proof_shape.starting_tidx[*idx]);
                cols.starting_cidx = F::from_usize(cidx);
                let has_preprocessed = child_vk.inner.per_air[*idx].preprocessed_data.is_some();
                cidx += has_preprocessed as usize;
                cols.n_logup = F::from_usize(preflight.proof_shape.n_logup);

                cols.is_present = F::ONE;
                cols.height = F::from_usize(height);
                cols.num_present = F::from_usize(num_present);

                let lifted_height = height.max(1 << l_skip);
                let num_interactions_per_row = child_vk.inner.per_air[*idx].num_interactions();
                let num_interactions = num_interactions_per_row * lifted_height;
                let lifted_height_limbs = decompose_usize::<NUM_LIMBS, LIMB_BITS>(lifted_height);
                let num_interactions_limbs =
                    decompose_usize::<NUM_LIMBS, LIMB_BITS>(num_interactions);
                cols.lifted_height_limbs = lifted_height_limbs.map(F::from_usize);
                cols.num_interactions_limbs = num_interactions_limbs.map(F::from_usize);
                cols.total_interactions_limbs =
                    decompose_f::<F, NUM_LIMBS, LIMB_BITS>(total_interactions);
                total_interactions += num_interactions;

                cols.n_max = F::from_usize(preflight.proof_shape.n_max);
                cols.num_air_id_lookups = F::from_usize(bc_air_shape_lookups[*idx]);

                let vcols: &mut ProofShapeVarColsMut<'_, F> = &mut borrow_var_cols_mut(
                    &mut chunk[cols_width..],
                    idx_encoder.width(),
                    max_cached,
                );

                for (i, flag) in idx_encoder
                    .get_flag_pt(*idx)
                    .iter()
                    .map(|x| F::from_u32(*x))
                    .enumerate()
                {
                    vcols.idx_flags[i] = flag;
                }

                for (i, commit) in vdata.cached_commitments.iter().enumerate() {
                    vcols.cached_commits[i] = *commit;
                    cidx += 1;
                }

                if *idx == min_cached_idx {
                    vcols.cached_commits[max_cached - 1] = proof.common_main_commit;
                }

                let next_total_interactions =
                    decompose_usize::<NUM_LIMBS, LIMB_BITS>(total_interactions);
                for i in 0..NUM_LIMBS {
                    range_checker.add_count(lifted_height_limbs[i]);
                    range_checker.add_count(next_total_interactions[i]);
                }

                let (nonzero_idx, height_limb) = lifted_height_limbs
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|&(_, limb)| limb != 0)
                    .unwrap();

                let mut carry = 0;
                let interactions_per_row_limbs =
                    decompose_usize::<NUM_LIMBS, LIMB_BITS>(num_interactions_per_row);
                // carry is 0 for i in 0..nonzero_idx
                range_checker.add_count_mult(0, nonzero_idx as u32);
                for i in nonzero_idx..NUM_LIMBS - 1 {
                    carry += height_limb * interactions_per_row_limbs[i - nonzero_idx];
                    carry = (carry - num_interactions_limbs[i]) >> LIMB_BITS;
                    range_checker.add_count(carry);
                }

                if sorted_idx < preflight.proof_shape.sorted_trace_vdata.len() {
                    let diff = vdata.log_height
                        - preflight.proof_shape.sorted_trace_vdata[sorted_idx]
                            .1
                            .log_height;
                    pow_checker.add_range(diff);
                } else if sorted_idx < num_airs {
                    pow_checker.add_range(log_height);
                }
                pow_checker.add_range(n.unsigned_abs());
                pow_checker.add_pow(log_height);
            }

            let total_interactions_f = decompose_f::<F, NUM_LIMBS, LIMB_BITS>(total_interactions);
            let total_interactions_usize =
                decompose_usize::<NUM_LIMBS, LIMB_BITS>(total_interactions);
            let num_present = F::from_usize(num_present);

            // Non-present AIRs
            for idx in (0..num_airs).filter(|idx| proof.trace_vdata[*idx].is_none()) {
                let chunk = chunks.next().unwrap();
                let cols: &mut ProofShapeCols<F, NUM_LIMBS> = chunk[..cols_width].borrow_mut();

                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(sorted_idx == 0);

                cols.sorted_idx = F::from_usize(sorted_idx);
                sorted_idx += 1;

                cols.num_present = num_present;

                cols.starting_tidx = F::from_usize(preflight.proof_shape.starting_tidx[idx]);
                cols.starting_cidx = F::from_usize(cidx);
                cols.n_logup = F::from_usize(preflight.proof_shape.n_logup);

                cols.total_interactions_limbs = total_interactions_f;
                cols.n_max = F::from_usize(preflight.proof_shape.n_max);
                cols.num_air_id_lookups = F::from_usize(bc_air_shape_lookups[idx]);

                let vcols: &mut ProofShapeVarColsMut<'_, F> = &mut borrow_var_cols_mut(
                    &mut chunk[cols_width..],
                    idx_encoder.width(),
                    max_cached,
                );

                for (i, flag) in idx_encoder
                    .get_flag_pt(idx)
                    .iter()
                    .map(|x| F::from_u32(*x))
                    .enumerate()
                {
                    vcols.idx_flags[i] = flag;
                }

                if idx == min_cached_idx {
                    vcols.cached_commits[max_cached - 1] = proof.common_main_commit;
                }

                range_checker.add_count_mult(0, (2 * NUM_LIMBS - 1) as u32);
                for limb in total_interactions_usize {
                    range_checker.add_count(limb);
                }

                if sorted_idx < num_airs {
                    pow_checker.add_range(0);
                }
            }

            debug_assert_eq!(num_airs, sorted_idx);

            // Summary row
            {
                let chunk = chunks.next().unwrap();
                let cols: &mut ProofShapeCols<F, NUM_LIMBS> = chunk[..cols_width].borrow_mut();

                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_last = F::ONE;
                cols.starting_tidx = F::from_usize(preflight.proof_shape.post_tidx);
                cols.num_present = num_present;

                let n_logup = preflight.proof_shape.n_logup;
                debug_assert_eq!(
                    u32::try_from(total_interactions).unwrap().leading_zeros(),
                    if total_interactions == 0 {
                        u32::BITS
                    } else {
                        u32::BITS - (l_skip + n_logup) as u32
                    }
                );
                let (nonzero_idx, has_interactions) = (0..NUM_LIMBS)
                    .rev()
                    .find(|&i| total_interactions_f[i] != F::ZERO)
                    .map(|idx| (idx, true))
                    .unwrap_or((0, false));
                let msb_limb = total_interactions_f[nonzero_idx];
                tracing::debug!(%l_skip, %n_logup, %total_interactions, %nonzero_idx, %msb_limb);
                let msb_limb_zero_bits = if has_interactions {
                    let msb_limb_num_bits = u32::BITS - msb_limb.as_canonical_u32().leading_zeros();
                    LIMB_BITS - msb_limb_num_bits as usize
                } else {
                    0
                };

                // non_zero_marker
                cols.lifted_height_limbs = from_fn(|i| {
                    if i == nonzero_idx && has_interactions {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                });
                // limb_to_range_check
                cols.height = msb_limb;
                // limb_to_range_check_inv
                cols.n_sign_bit = if msb_limb != F::ZERO {
                    msb_limb.inverse()
                } else {
                    F::ZERO
                };
                // msb_limb_zero_bits_exp
                cols.log_height = F::from_usize(1 << msb_limb_zero_bits);

                let max_interactions = decompose_f::<F, NUM_LIMBS, LIMB_BITS>(
                    child_vk.inner.params.logup.max_interaction_count as usize,
                );
                let diff_idx = (0..NUM_LIMBS)
                    .rev()
                    .find(|&i| total_interactions_f[i] != max_interactions[i])
                    .unwrap_or(0);

                // diff_marker
                cols.num_interactions_limbs =
                    from_fn(|i| if i == diff_idx { F::ONE } else { F::ZERO });

                cols.total_interactions_limbs = total_interactions_f;
                cols.n_max = F::from_usize(preflight.proof_shape.n_max);
                cols.is_n_max_greater = F::from_bool(preflight.proof_shape.n_max > n_logup);
                cols.n_logup = F::from_usize(n_logup);

                let shifted_msb_limb =
                    msb_limb.as_canonical_u32() as usize * (1 << msb_limb_zero_bits);
                range_checker.add_count(shifted_msb_limb);
                if shifted_msb_limb != 0 {
                    range_checker.add_count(shifted_msb_limb - (1 << (LIMB_BITS - 1)));
                }

                range_checker.add_count(
                    (max_interactions[diff_idx] - total_interactions_f[diff_idx]).as_canonical_u32()
                        as usize
                        - 1,
                );

                pow_checker.add_pow(msb_limb_zero_bits);
                pow_checker.add_range(preflight.proof_shape.n_max.abs_diff(n_logup));

                // We store the pre-hash of the child vk in the summary row
                let vcols: &mut ProofShapeVarColsMut<'_, F> = &mut borrow_var_cols_mut(
                    &mut chunk[cols_width..],
                    idx_encoder.width(),
                    max_cached,
                );
                vcols.cached_commits[max_cached - 1] = child_vk.pre_hash;
            }
        }

        for chunk in chunks {
            let cols: &mut ProofShapeCols<F, NUM_LIMBS> = chunk[..cols_width].borrow_mut();
            cols.proof_idx = F::from_usize(proofs.len());
        }

        Some(RowMajorMatrix::new(trace, total_width))
    }
}
