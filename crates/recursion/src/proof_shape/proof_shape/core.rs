use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use itertools::fold;
use openvm_circuit_primitives::{
    SubAir,
    encoder::Encoder,
    utils::{and, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{DIGEST_SIZE, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        AirHeightsBus, AirHeightsBusMessage, AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus,
        AirShapeBusMessage, AirShapeProperty, CommitmentsBus, CommitmentsBusMessage, GkrModuleBus,
        GkrModuleMessage, TranscriptBus, TranscriptBusMessage,
    },
    primitives::{
        bus::{PowerCheckerBus, PowerCheckerBusMessage, RangeCheckerBus, RangeCheckerBusMessage},
        pow::PowerCheckerTraceGenerator,
        range::RangeCheckerTraceGenerator,
    },
    proof_shape::{
        AirMetadata,
        bus::{
            NumPublicValuesBus, NumPublicValuesMessage, ProofShapePermutationBus,
            ProofShapePermutationMessage,
        },
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ProofShapeCols<F, const NUM_LIMBS: usize> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    pub idx: F,
    pub sorted_idx: F,
    pub log_height: F,

    // Lookup multiplicities for AirPartShapeBus. Note that part_cached_mult provides the
    // multiplicities for cached traces.
    pub part_common_main_mult: F,
    pub part_preprocessed_mult: F,

    // First possible non-main cidx of the current AIR
    pub starting_cidx: F,

    // Columns that may be read from the transcript. Note that cached_commits is also read
    // from the transcript.
    pub is_present: F,
    pub height: F,

    // The total number of interactions over all traces needs to fit in a single field element,
    // so we assume that it only requires INTERACTIONS_LIMBS (4) limbs to store.
    //
    // To constrain the correctness of n_logup, we ensure that total_interactions_limbs has
    // CELLS_LIMBS * LIMB_BITS - n_logup leading zeroes. We do this by a) recording the most
    // significant non-zero limb i and b) making sure total_interaction_limbs[i] * 2^{the
    // number of remaining leading zeroes} is within [0, 256).
    //
    // To constrain that the total number of interactions over all traces is less than the
    // max interactions set in the vk, we record the most significant limb at which the max
    // limb decomposition and total_interactions_limbs differ. The difference between those
    // two limbs is then range checked to be within [1, 256).
    pub height_limbs: [F; NUM_LIMBS],
    pub num_interactions_limbs: [F; NUM_LIMBS],
    pub total_interactions_limbs: [F; NUM_LIMBS],

    pub n_max: F,
    pub is_n_max_greater: F,
}

// Variable-length columns are stored at the end
pub struct ProofShapeVarCols<'a, F> {
    pub idx_flags: &'a [F],                     // [F; IDX_FLAGS]
    pub part_cached_mult: &'a [F],              // [F; MAX_CACHED]
    pub cached_commits: &'a [[F; DIGEST_SIZE]], // [[F; DIGEST_SIZE]; MAX_CACHED]
}

pub struct ProofShapeVarColsMut<'a, F> {
    pub idx_flags: &'a mut [F],                     // [F; IDX_FLAGS]
    pub part_cached_mult: &'a mut [F],              // [F; MAX_CACHED]
    pub cached_commits: &'a mut [[F; DIGEST_SIZE]], // [[F; DIGEST_SIZE]; MAX_CACHED]
}

#[derive(derive_new::new)]
pub(in crate::proof_shape) struct ProofShapeChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    idx_encoder: Arc<Encoder>,
    min_cached_idx: usize,
    max_cached: usize,
    range_checker: Arc<RangeCheckerTraceGenerator<LIMB_BITS>>,
    pow_checker: Arc<PowerCheckerTraceGenerator<2, 32>>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> ProofShapeChip<NUM_LIMBS, LIMB_BITS> {
    pub(in crate::proof_shape) fn generate_trace(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> RowMajorMatrix<F> {
        let idx_encoder = &self.idx_encoder;
        let min_cached_idx = self.min_cached_idx;
        let max_cached = self.max_cached;
        let range_checker = &self.range_checker;
        let pow_checker = &self.pow_checker;
        let num_airs = child_vk.inner.per_air.len();
        let num_rows = (proofs.len() * (num_airs + 1)).next_power_of_two();
        let cols_width = ProofShapeCols::<usize, NUM_LIMBS>::width();
        let total_width =
            self.idx_encoder.width() + cols_width + self.max_cached * (DIGEST_SIZE + 1);

        debug_assert_eq!(proofs.len(), preflights.len());

        let mut trace = vec![F::ZERO; num_rows * total_width];
        let mut chunks = trace.chunks_exact_mut(total_width);

        for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights.iter()).enumerate() {
            let mut sorted_idx = 0usize;
            let mut total_interactions = 0usize;
            let mut cidx = 1usize;

            // Present AIRs
            for (idx, vdata) in &preflight.proof_shape.sorted_trace_vdata {
                let chunk = chunks.next().unwrap();
                let cols: &mut ProofShapeCols<F, NUM_LIMBS> = chunk[..cols_width].borrow_mut();
                let log_height = vdata.hypercube_dim + child_vk.inner.params.l_skip;
                let height = 1usize << log_height;

                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(sorted_idx == 0);

                cols.idx = F::from_canonical_usize(*idx);
                cols.sorted_idx = F::from_canonical_usize(sorted_idx);
                cols.log_height = F::from_canonical_usize(log_height);
                sorted_idx += 1;

                cols.starting_cidx = F::from_canonical_usize(cidx);
                let has_preprocessed = child_vk.inner.per_air[*idx].preprocessed_data.is_some();
                cidx += has_preprocessed as usize;

                // TODO[stephen]: these multiplicities might not be one
                cols.part_common_main_mult = F::ONE;
                cols.part_preprocessed_mult = F::from_bool(has_preprocessed);

                cols.is_present = F::ONE;
                cols.height = F::from_canonical_usize(height);

                let num_interactions = child_vk.inner.per_air[*idx].num_interactions() * height;
                let height_limbs = decompose_usize::<NUM_LIMBS, LIMB_BITS>(height);
                let num_interactions_limbs =
                    decompose_usize::<NUM_LIMBS, LIMB_BITS>(num_interactions);
                cols.height_limbs = height_limbs.map(F::from_canonical_usize);
                cols.num_interactions_limbs = num_interactions_limbs.map(F::from_canonical_usize);
                cols.total_interactions_limbs =
                    decompose_f::<F, NUM_LIMBS, LIMB_BITS>(total_interactions);
                total_interactions += num_interactions;

                cols.n_max = F::from_canonical_usize(preflight.proof_shape.n_max);

                let vcols: &mut ProofShapeVarColsMut<'_, F> = &mut borrow_var_cols_mut(
                    &mut chunk[cols_width..],
                    idx_encoder.width(),
                    max_cached,
                );

                for (i, flag) in idx_encoder
                    .get_flag_pt(*idx)
                    .iter()
                    .map(|x| F::from_canonical_u32(*x))
                    .enumerate()
                {
                    vcols.idx_flags[i] = flag;
                }

                for (i, commit) in vdata.cached_commitments.iter().enumerate() {
                    // TODO[stephen]: this multiplicities might not be one
                    vcols.part_cached_mult[i] = F::ONE;
                    vcols.cached_commits[i] = *commit;
                    cidx += 1;
                }

                if *idx == min_cached_idx {
                    vcols.cached_commits[max_cached - 1] = proof.common_main_commit;
                }

                let next_total_interactions =
                    decompose_usize::<NUM_LIMBS, LIMB_BITS>(total_interactions);
                for i in 0..NUM_LIMBS {
                    range_checker.add_count(height_limbs[i]);
                    range_checker.add_count(next_total_interactions[i]);
                }

                let (nonzero_idx, height_limb) = height_limbs
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|&(_, limb)| limb != 0)
                    .unwrap();

                for limb in num_interactions_limbs
                    .iter()
                    .take(NUM_LIMBS - 1)
                    .skip(nonzero_idx)
                {
                    range_checker.add_count((height_limb * limb) >> LIMB_BITS);
                }
                range_checker.add_count_mult(0, nonzero_idx as u32);

                if sorted_idx < preflight.proof_shape.sorted_trace_vdata.len() {
                    pow_checker.add_range(
                        vdata.hypercube_dim
                            - preflight.proof_shape.sorted_trace_vdata[sorted_idx]
                                .1
                                .hypercube_dim,
                    );
                } else if sorted_idx < num_airs {
                    pow_checker.add_range(log_height);
                }
                pow_checker.add_pow(log_height);
            }

            let total_interactions_f = decompose_f::<F, NUM_LIMBS, LIMB_BITS>(total_interactions);
            let total_interactions_usize =
                decompose_usize::<NUM_LIMBS, LIMB_BITS>(total_interactions);

            // Non-present AIRs
            for idx in (0..num_airs).filter(|idx| proof.trace_vdata[*idx].is_none()) {
                let chunk = chunks.next().unwrap();
                let cols: &mut ProofShapeCols<F, NUM_LIMBS> = chunk[..cols_width].borrow_mut();

                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(sorted_idx == 0);

                cols.idx = F::from_canonical_usize(idx);
                cols.sorted_idx = F::from_canonical_usize(sorted_idx);
                sorted_idx += 1;

                cols.starting_cidx = F::from_canonical_usize(cidx);

                cols.total_interactions_limbs = total_interactions_f;
                cols.n_max = F::from_canonical_usize(preflight.proof_shape.n_max);

                let vcols: &mut ProofShapeVarColsMut<'_, F> = &mut borrow_var_cols_mut(
                    &mut chunk[cols_width..],
                    idx_encoder.width(),
                    max_cached,
                );

                for (i, flag) in idx_encoder
                    .get_flag_pt(idx)
                    .iter()
                    .map(|x| F::from_canonical_u32(*x))
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

                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.is_last = F::ONE;

                let (nonzero_idx, has_interactions) = (0..NUM_LIMBS)
                    .rev()
                    .find(|&i| total_interactions_f[i] != F::ZERO)
                    .map(|idx| (idx, true))
                    .unwrap_or((0, false));
                let msb_limb = total_interactions_f[nonzero_idx];
                let msb_limb_zero_bits = if has_interactions {
                    let msb_limb_num_bits = u32::BITS - msb_limb.as_canonical_u32().leading_zeros();
                    LIMB_BITS - msb_limb_num_bits as usize
                } else {
                    0
                };

                // non_zero_marker
                cols.height_limbs = from_fn(|i| {
                    if i == nonzero_idx && has_interactions {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                });
                // limb_to_range_check
                cols.height = msb_limb;
                // msb_limb_zero_bits_exp
                cols.log_height = F::from_canonical_usize(1 << msb_limb_zero_bits);

                let max_interactions =
                    decompose_f::<F, NUM_LIMBS, LIMB_BITS>(F::ORDER_U32 as usize);
                let diff_idx = (0..NUM_LIMBS)
                    .rev()
                    .find(|&i| total_interactions_f[i] != max_interactions[i])
                    .unwrap_or(0);

                // diff_marker
                cols.num_interactions_limbs =
                    from_fn(|i| if i == diff_idx { F::ONE } else { F::ZERO });

                cols.total_interactions_limbs = total_interactions_f;
                cols.n_max = F::from_canonical_usize(preflight.proof_shape.n_max);
                cols.is_n_max_greater =
                    F::from_bool(preflight.proof_shape.n_max > preflight.proof_shape.n_logup);

                // n_logup
                cols.starting_cidx = F::from_canonical_usize(preflight.proof_shape.n_logup);

                range_checker
                    .add_count(msb_limb.as_canonical_u32() as usize * (1 << msb_limb_zero_bits));
                range_checker.add_count(
                    (max_interactions[diff_idx] - total_interactions_f[diff_idx]).as_canonical_u32()
                        as usize
                        - 1,
                );

                pow_checker.add_pow(msb_limb_zero_bits);
                pow_checker.add_range(
                    preflight
                        .proof_shape
                        .n_max
                        .abs_diff(preflight.proof_shape.n_logup),
                );

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
            cols.proof_idx = F::from_canonical_usize(proofs.len());
        }

        RowMajorMatrix::new(trace, total_width)
    }
}

pub struct ProofShapeAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    // Parameters derived from vk
    pub per_air: Vec<AirMetadata>,
    pub l_skip: usize,
    pub min_cached_idx: usize,
    pub max_cached: usize,
    pub commit_mult: usize,

    // Primitives
    pub idx_encoder: Arc<Encoder>,
    pub range_bus: RangeCheckerBus,
    pub pow_bus: PowerCheckerBus,

    // Internal buses
    pub permutation_bus: ProofShapePermutationBus,
    pub num_pvs_bus: NumPublicValuesBus,

    // Inter-module buses
    pub gkr_module_bus: GkrModuleBus,
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub air_heights_bus: AirHeightsBus,
    pub commitments_bus: CommitmentsBus,
    pub transcript_bus: TranscriptBus,
}

impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ProofShapeAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ProofShapeCols::<F, NUM_LIMBS>::width()
            + self.idx_encoder.width()
            + self.max_cached * (DIGEST_SIZE + 1)
    }
}
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for ProofShapeAir<NUM_LIMBS, LIMB_BITS>
{
}
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> PartitionedBaseAir<F>
    for ProofShapeAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, AB: AirBuilder + InteractionBuilder> Air<AB>
    for ProofShapeAir<NUM_LIMBS, LIMB_BITS>
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let const_width = ProofShapeCols::<AB::Var, NUM_LIMBS>::width();

        let localv = borrow_var_cols::<AB::Var>(
            &local[const_width..],
            self.idx_encoder.width(),
            self.max_cached,
        );
        let local: &ProofShapeCols<AB::Var, NUM_LIMBS> = (*local)[..const_width].borrow();
        let next: &ProofShapeCols<AB::Var, NUM_LIMBS> = (*next)[..const_width].borrow();

        self.idx_encoder.eval(builder, localv.idx_flags);

        NestedForLoopSubAir::<1, 0> {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid + local.is_last,
                        counter: [local.proof_idx.into()],
                        is_first: [local.is_first.into()],
                    },
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid + next.is_last,
                        counter: [next.proof_idx.into()],
                        is_first: [next.is_first.into()],
                    },
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );
        builder
            .when(and(local.is_valid, not(local.is_last)))
            .assert_eq(local.proof_idx, next.proof_idx);

        builder.assert_bool(local.is_present);
        builder.when(local.is_present).assert_one(local.is_valid);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // PERMUTATION AND SORTING
        ///////////////////////////////////////////////////////////////////////////////////////////
        builder.when(local.is_first).assert_zero(local.sorted_idx);
        builder
            .when(next.sorted_idx)
            .assert_eq(local.sorted_idx, next.sorted_idx - AB::F::ONE);

        self.permutation_bus.send(
            builder,
            local.proof_idx,
            ProofShapePermutationMessage {
                idx: local.sorted_idx,
            },
            local.is_valid,
        );

        self.permutation_bus.receive(
            builder,
            local.proof_idx,
            ProofShapePermutationMessage { idx: local.idx },
            local.is_valid,
        );

        builder
            .when(and(not(local.is_present), local.is_valid))
            .assert_zero(local.height);

        self.pow_bus.lookup_key(
            builder,
            PowerCheckerBusMessage {
                log: local.log_height,
                exp: local.height,
            },
            local.is_present,
        );

        // Range check difference using ExponentBus to ensure local.log_height >= next.log_height
        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: local.log_height - next.log_height,
                max_bits: AB::Expr::from_canonical_usize(5),
            },
            and(local.is_valid, not(next.is_last)),
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // VK FIELD SELECTION
        ///////////////////////////////////////////////////////////////////////////////////////////
        let mut tidx_offset = 2 * DIGEST_SIZE;
        let mut tidx = AB::Expr::ZERO;
        let mut num_interactions_per_row = [AB::Expr::ZERO; NUM_LIMBS];

        // Select values for TranscriptBus
        let mut is_required = AB::Expr::ZERO;
        let mut is_min_cached = AB::Expr::ZERO;
        let mut has_preprocessed = AB::Expr::ZERO;
        let mut cached_present = vec![AB::Expr::ZERO; self.max_cached];

        // Select values for AirShapeBus
        let mut num_interactions = AB::Expr::ZERO;

        // Select values for AirPartShapeBus
        let mut main_common_width = AB::Expr::ZERO;
        let mut preprocessed_stacked_width = AB::Expr::ZERO;
        let mut cached_widths = vec![AB::Expr::ZERO; self.max_cached];

        // Select values for CommitmentsBus
        let mut preprocessed_commit = [AB::Expr::ZERO; DIGEST_SIZE];

        // Select values for NumPublicValuesBus
        let mut num_pvs = AB::Expr::ZERO;
        let mut has_pvs = AB::Expr::ZERO;

        for (i, air_data) in self.per_air.iter().enumerate() {
            // We keep a running tally of how many transcript reads there should be up to any
            // given point, and use that to constrain initial_tidx
            let is_current_air = self.idx_encoder.get_flag_expr::<AB>(i, localv.idx_flags);
            let mut when_current = builder.when(is_current_air.clone());

            when_current.assert_eq(local.idx, AB::F::from_canonical_usize(i));

            tidx += is_current_air.clone() * AB::F::from_canonical_usize(tidx_offset);

            main_common_width +=
                is_current_air.clone() * AB::F::from_canonical_usize(air_data.main_width);

            if air_data.num_public_values != 0 {
                has_pvs += is_current_air.clone();
            }
            num_pvs +=
                is_current_air.clone() * AB::F::from_canonical_usize(air_data.num_public_values);

            // Select number of interactions for use later in the AIR and constrain that the
            // num_interactions_per_row limb decomposition is correct.
            num_interactions +=
                is_current_air.clone() * AB::F::from_canonical_usize(air_data.num_interactions);

            for (i, &limb) in decompose_f::<AB::F, NUM_LIMBS, LIMB_BITS>(air_data.num_interactions)
                .iter()
                .enumerate()
            {
                num_interactions_per_row[i] += is_current_air.clone() * limb;
            }

            if air_data.is_required {
                is_required += is_current_air.clone();
                when_current.assert_one(local.is_present);
            } else {
                tidx_offset += 1;
            }

            if i == self.min_cached_idx {
                is_min_cached += is_current_air.clone();
                when_current.assert_zero(localv.part_cached_mult[self.max_cached - 1]);
            }

            if let Some(preprocessed) = &air_data.preprocessed_data {
                when_current.assert_eq(
                    local.log_height,
                    AB::Expr::from_canonical_usize(preprocessed.hypercube_dim + self.l_skip),
                );
                has_preprocessed += is_current_air.clone();

                preprocessed_stacked_width += is_current_air.clone()
                    * AB::F::from_canonical_usize(air_data.preprocessed_width.unwrap());
                preprocessed_commit = from_fn(|i| {
                    is_current_air.clone()
                        * AB::F::from_canonical_u32(preprocessed.commit[i].as_canonical_u32())
                });

                tidx_offset += DIGEST_SIZE;
            } else {
                tidx_offset += 1;
            }

            for (cached_idx, width) in air_data.cached_widths.iter().enumerate() {
                cached_present[cached_idx] += is_current_air.clone();
                cached_widths[cached_idx] +=
                    is_current_air.clone() * AB::Expr::from_canonical_usize(*width);
                tidx_offset += DIGEST_SIZE;
            }

            tidx_offset += air_data.num_public_values;
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // TRANSCRIPT OBSERVATIONS
        ///////////////////////////////////////////////////////////////////////////////////////////
        let hypercube_dim = local.log_height.into() - AB::F::from_canonical_usize(self.l_skip);

        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: tidx.clone(),
                value: local.is_present.into(),
                is_sample: AB::Expr::ZERO,
            },
            not::<AB::Expr>(is_required.clone()) * local.is_valid,
        );
        tidx += not::<AB::Expr>(is_required);

        for (didx, commit_val) in preprocessed_commit.iter().enumerate() {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_canonical_usize(didx),
                    value: commit_val.clone(),
                    is_sample: AB::Expr::ZERO,
                },
                has_preprocessed.clone() * local.is_valid,
            );
        }
        tidx += has_preprocessed.clone() * AB::Expr::from_canonical_usize(DIGEST_SIZE);

        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: tidx.clone(),
                value: hypercube_dim.clone(),
                is_sample: AB::Expr::ZERO,
            },
            not::<AB::Expr>(has_preprocessed.clone()) * local.is_valid,
        );
        tidx += not::<AB::Expr>(has_preprocessed.clone());

        (0..self.max_cached).for_each(|i| {
            for didx in 0..DIGEST_SIZE {
                self.transcript_bus.receive(
                    builder,
                    local.proof_idx,
                    TranscriptBusMessage {
                        tidx: tidx.clone(),
                        value: localv.cached_commits[i][didx].into(),
                        is_sample: AB::Expr::ZERO,
                    },
                    cached_present[i].clone() * local.is_valid,
                );
                tidx += cached_present[i].clone();
            }
        });

        let num_pvs_tidx = tidx.clone();
        tidx += num_pvs.clone();

        for didx in 0..DIGEST_SIZE {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(didx),
                    value: localv.cached_commits[self.max_cached - 1][didx].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_last,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(didx + DIGEST_SIZE),
                    value: localv.cached_commits[self.max_cached - 1][didx].into(),
                    is_sample: AB::Expr::ZERO,
                },
                is_min_cached.clone() * local.is_valid,
            );
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // AIR SHAPE LOOKUP
        ///////////////////////////////////////////////////////////////////////////////////////////
        let num_main_parts = fold(cached_present.iter(), AB::Expr::ONE, |acc, is_present| {
            acc + is_present.clone()
        });

        self.air_shape_bus.send(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sorted_idx.into(),
                property_idx: AirShapeProperty::AirId.to_field(),
                value: local.idx.into(),
            },
            local.is_present,
        );
        self.air_shape_bus.send(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sorted_idx.into(),
                property_idx: AirShapeProperty::HypercubeDim.to_field(),
                value: hypercube_dim.clone(),
            },
            local.is_present,
        );
        self.air_shape_bus.send(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sorted_idx.into(),
                property_idx: AirShapeProperty::HasPreprocessed.to_field(),
                value: has_preprocessed.clone(),
            },
            local.is_present,
        );
        self.air_shape_bus.send(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sorted_idx.into(),
                property_idx: AirShapeProperty::NumMainParts.to_field(),
                value: num_main_parts,
            },
            local.is_present,
        );
        self.air_shape_bus.send(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sorted_idx.into(),
                property_idx: AirShapeProperty::NumInteractions.to_field(),
                value: num_interactions,
            },
            local.is_present,
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // AIR PART SHAPE LOOKUP
        ///////////////////////////////////////////////////////////////////////////////////////////
        builder
            .when(not::<AB::Expr>(local.is_present))
            .assert_zero(local.part_common_main_mult);
        builder
            .when(not::<AB::Expr>(has_preprocessed.clone()))
            .assert_zero(local.part_preprocessed_mult);
        (0..self.max_cached).for_each(|cached_idx| {
            builder
                .when(not::<AB::Expr>(cached_present[cached_idx].clone()))
                .assert_zero(localv.part_cached_mult[cached_idx]);
        });

        self.air_part_shape_bus.send(
            builder,
            local.proof_idx,
            AirPartShapeBusMessage {
                idx: local.idx.into(),
                part: AB::Expr::ZERO,
                width: main_common_width.clone(),
            },
            local.part_common_main_mult * local.is_valid,
        );

        self.air_part_shape_bus.send(
            builder,
            local.proof_idx,
            AirPartShapeBusMessage {
                idx: local.idx.into(),
                part: AB::Expr::ONE,
                width: preprocessed_stacked_width.clone(),
            },
            local.part_preprocessed_mult * local.is_valid,
        );

        (0..self.max_cached).for_each(|cached_idx| {
            self.air_part_shape_bus.send(
                builder,
                local.proof_idx,
                AirPartShapeBusMessage {
                    idx: local.idx.into(),
                    part: AB::Expr::from_canonical_usize(1 + cached_idx) + has_preprocessed.clone(),
                    width: cached_widths[cached_idx].clone(),
                },
                localv.part_cached_mult[cached_idx] * local.is_valid,
            );
        });

        ///////////////////////////////////////////////////////////////////////////////////////////
        // AIR HEIGHTS LOOKUP
        ///////////////////////////////////////////////////////////////////////////////////////////
        let total_width = fold(
            cached_widths,
            main_common_width + preprocessed_stacked_width,
            |acc, width| acc + width,
        );
        self.air_heights_bus.send(
            builder,
            local.proof_idx,
            AirHeightsBusMessage {
                sort_idx: local.sorted_idx,
                height: local.height,
                log_height: local.log_height,
            },
            local.is_present * total_width,
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // STACKING COMMITMENTS
        ///////////////////////////////////////////////////////////////////////////////////////////
        let mut cidx = local.starting_cidx.into();
        builder
            .when(and(local.is_first, local.is_valid))
            .assert_one(cidx.clone());

        self.commitments_bus.send(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: AB::Expr::ZERO,
                minor_idx: cidx.clone(),
                commitment: preprocessed_commit,
            },
            has_preprocessed.clone()
                * local.is_valid
                * AB::Expr::from_canonical_usize(self.commit_mult),
        );

        cidx += has_preprocessed.clone();
        (0..self.max_cached).for_each(|cached_idx| {
            self.commitments_bus.send(
                builder,
                local.proof_idx,
                CommitmentsBusMessage {
                    major_idx: AB::Expr::ZERO,
                    minor_idx: cidx.clone(),
                    commitment: localv.cached_commits[cached_idx].map(Into::into),
                },
                cached_present[cached_idx].clone()
                    * local.is_valid
                    * AB::Expr::from_canonical_usize(self.commit_mult),
            );
            cidx += cached_present[cached_idx].clone();
        });

        builder
            .when(and(local.is_valid, not(next.is_last)))
            .assert_eq(cidx, next.starting_cidx);

        self.commitments_bus.send(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: AB::Expr::ZERO,
                minor_idx: AB::Expr::ZERO,
                commitment: localv.cached_commits[self.max_cached - 1].map(Into::into),
            },
            is_min_cached.clone()
                * local.is_valid
                * AB::Expr::from_canonical_usize(self.commit_mult),
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // NUM PUBLIC VALUES
        ///////////////////////////////////////////////////////////////////////////////////////////
        self.num_pvs_bus.send(
            builder,
            local.proof_idx,
            NumPublicValuesMessage {
                air_idx: local.idx.into(),
                tidx: num_pvs_tidx,
                num_pvs,
            },
            local.is_present * has_pvs,
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // INTERACTIONS + GKR MESSAGE
        ///////////////////////////////////////////////////////////////////////////////////////////
        // Constrain that height decomposition is correct. Note we constrained the width
        // decomposition to be correct above.
        builder.when(local.is_valid).assert_eq(
            fold(
                local.height_limbs.iter().enumerate(),
                AB::Expr::ZERO,
                |acc, (i, limb)| acc + (AB::Expr::from_canonical_u32(1 << (i * LIMB_BITS)) * *limb),
            ),
            local.height,
        );

        for i in 0..NUM_LIMBS {
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: local.height_limbs[i].into(),
                    max_bits: AB::Expr::from_canonical_usize(LIMB_BITS),
                },
                local.is_valid,
            );
        }

        // Constrain that num_interactions = height * num_interactions_per_row
        let mut carry = vec![AB::Expr::ZERO; NUM_LIMBS * 2];
        let carry_divide = AB::F::from_canonical_u32(1 << LIMB_BITS).inverse();

        for (i, &height_limb) in local.height_limbs.iter().enumerate() {
            for (j, interactions_limb) in num_interactions_per_row.iter().enumerate() {
                carry[i + j] += height_limb * interactions_limb.clone();
            }
        }

        for i in 0..2 * NUM_LIMBS {
            if i != 0 {
                let prev = carry[i - 1].clone();
                carry[i] += prev;
            }
            carry[i] = AB::Expr::from(carry_divide)
                * (carry[i].clone()
                    - if i < NUM_LIMBS {
                        local.num_interactions_limbs[i].into()
                    } else {
                        AB::Expr::ZERO
                    });
            if i < NUM_LIMBS - 1 {
                self.range_bus.lookup_key(
                    builder,
                    RangeCheckerBusMessage {
                        value: carry[i].clone(),
                        max_bits: AB::Expr::from_canonical_usize(LIMB_BITS),
                    },
                    local.is_valid,
                );
            } else {
                builder.when(local.is_valid).assert_zero(carry[i].clone());
            }
        }

        // Constrain total number of interactions is added correctly. For induction, we must also
        // constrain that the initial total number of interactions is zero.
        local.total_interactions_limbs.iter().for_each(|x| {
            builder.when(local.is_first).assert_zero(*x);
        });

        for i in 0..NUM_LIMBS {
            carry[i] = AB::Expr::from(carry_divide)
                * (local.num_interactions_limbs[i].into() + local.total_interactions_limbs[i]
                    - next.total_interactions_limbs[i]
                    + if i > 0 {
                        carry[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            if i < NUM_LIMBS - 1 {
                builder.when(local.is_valid).assert_bool(carry[i].clone());
            } else {
                builder.when(local.is_valid).assert_zero(carry[i].clone());
            }
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: next.total_interactions_limbs[i].into(),
                    max_bits: AB::Expr::from_canonical_usize(LIMB_BITS),
                },
                local.is_valid,
            );
        }

        // While the (N + 1)-th row (index N) is invalid, we use it to store the final number
        // of total cells. We thus can always constrain local.total_cells + local.num_cells =
        // next.total_cells when local is valid, and when we're on the summary row we can send
        // the stacking main width message.
        //
        // Note that we must constrain that the is_last flag is set correctly, i.e. it must
        // only be set on the row immediately after the N-th.
        builder.assert_bool(local.is_last);
        builder.when(local.is_last).assert_zero(local.is_valid);
        builder.when(next.is_last).assert_one(local.is_valid);
        builder
            .when(local.sorted_idx - AB::F::from_canonical_usize(self.per_air.len() - 1))
            .assert_zero(next.is_last);
        builder
            .when(next.is_last)
            .assert_zero(local.sorted_idx - AB::F::from_canonical_usize(self.per_air.len() - 1));

        // Constrain that n_logup is correct, i.e. that there are CELLS_LIMBS * LIMB_BITS - n_logup
        // leading zeroes in total_interactions_limbs. Because we only do this on the is_last row,
        // we can re-use several of our columns to save space.
        //
        // We mark the most significant non-zero limb of local.total_interactions_limbs using the
        // non_zero_marker column array defined below, and the remaining number of leading 0 bits
        // needed within the limb using msb_limb_zero_bits_exp. Column limb_to_range_check is used
        // to store the value of the most significant limb to range check.
        let non_zero_marker = local.height_limbs;
        let limb_to_range_check = local.height;
        let msb_limb_zero_bits_exp = local.log_height;
        let n_logup = local.starting_cidx;

        let mut prefix = AB::Expr::ZERO;
        let mut expected_limb_to_range_check = AB::Expr::ZERO;
        let mut msb_limb_zero_bits = AB::Expr::ZERO;

        for i in (0..NUM_LIMBS).rev() {
            prefix += non_zero_marker[i].into();
            expected_limb_to_range_check += local.total_interactions_limbs[i] * non_zero_marker[i];
            msb_limb_zero_bits +=
                non_zero_marker[i] * AB::F::from_canonical_usize((i + 1) * LIMB_BITS);

            builder.when(local.is_last).assert_bool(non_zero_marker[i]);
            builder
                .when(not::<AB::Expr>(prefix.clone()) * local.is_last)
                .assert_zero(local.total_interactions_limbs[i]);
            builder
                .when(local.total_interactions_limbs[i] * local.is_last)
                .assert_one(prefix.clone());
        }

        builder.when(local.is_last).assert_bool(prefix.clone());
        builder
            .when(local.is_last)
            .assert_eq(limb_to_range_check, expected_limb_to_range_check);
        msb_limb_zero_bits -= n_logup + prefix * AB::F::from_canonical_usize(self.l_skip);

        self.pow_bus.lookup_key(
            builder,
            PowerCheckerBusMessage {
                log: msb_limb_zero_bits,
                exp: msb_limb_zero_bits_exp.into(),
            },
            local.is_last,
        );

        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: limb_to_range_check * msb_limb_zero_bits_exp,
                max_bits: AB::Expr::from_canonical_usize(LIMB_BITS),
            },
            local.is_last,
        );

        // Constrain n_max on each row. Also constrain that local.is_n_max_greater is one when
        // n_max is greater than n_logup, and zero otherwise.
        builder
            .when(local.is_first)
            .assert_eq(local.n_max, hypercube_dim);
        builder
            .when(local.is_valid)
            .assert_eq(local.n_max, next.n_max);

        builder.assert_bool(local.is_n_max_greater);
        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: (local.n_max - n_logup) * (local.is_n_max_greater * AB::F::TWO - AB::F::ONE),
                max_bits: AB::Expr::from_canonical_usize(5),
            },
            local.is_last,
        );

        self.gkr_module_bus.send(
            builder,
            local.proof_idx,
            GkrModuleMessage {
                tidx: AB::Expr::from_canonical_usize(tidx_offset),
                n_logup: n_logup.into(),
                n_max: local.n_max.into(),
                n_global: local.is_n_max_greater * local.n_max
                    + not(local.is_n_max_greater) * n_logup,
            },
            local.is_last,
        );

        // Constrain that the total number of interactions is less than the vk-specified amount.
        // Once again we only do this on the is_last row. Column array diff_marker marks the most
        // significant non-zero limb where local.total_interactions_limbs and the decomposed
        // max_interactions_limbs (denoted p) differ. To constrain that total_interaction_limbs[i]
        // < p[i], we range check p[i] - total_interaction_limbs[i] - 1. Note that both p[i] and
        // total_interaction_limbs[i] are guaranteed to be in [0, 256), so it's impossible to have
        // p[i] - total_interaction_limbs[i] == 256.
        let diff_marker = local.num_interactions_limbs;

        let max_interactions =
            decompose_f::<AB::Expr, NUM_LIMBS, LIMB_BITS>(AB::F::ORDER_U32 as usize);
        let mut prefix = AB::Expr::ZERO;
        let mut diff_val = AB::Expr::ZERO;

        for i in (0..NUM_LIMBS).rev() {
            prefix += diff_marker[i].into();
            diff_val += diff_marker[i].into()
                * (max_interactions[i].clone() - local.total_interactions_limbs[i]);

            builder.when(local.is_last).assert_bool(diff_marker[i]);
            builder
                .when(not::<AB::Expr>(prefix.clone()) * local.is_last)
                .assert_zero(local.total_interactions_limbs[i]);
            builder
                .when(local.total_interactions_limbs[i] * local.is_last)
                .assert_one(prefix.clone());
        }

        builder.when(local.is_last).assert_one(prefix.clone());
        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: diff_val - AB::Expr::ONE,
                max_bits: AB::Expr::from_canonical_usize(LIMB_BITS),
            },
            local.is_last,
        );
    }
}

fn decompose_f<F: FieldAlgebra, const LIMBS: usize, const LIMB_BITS: usize>(
    value: usize,
) -> [F; LIMBS] {
    from_fn(|i| F::from_canonical_usize((value >> (i * LIMB_BITS)) & ((1 << LIMB_BITS) - 1)))
}

fn decompose_usize<const LIMBS: usize, const LIMB_BITS: usize>(value: usize) -> [usize; LIMBS] {
    from_fn(|i| (value >> (i * LIMB_BITS)) & ((1 << LIMB_BITS) - 1))
}

fn borrow_var_cols<F>(
    slice: &[F],
    idx_flags: usize,
    max_cached: usize,
) -> ProofShapeVarCols<'_, F> {
    let flags_idx = 0;
    let part_cached_idx = flags_idx + idx_flags;
    let cached_commits_idx = part_cached_idx + max_cached;

    let cached_commits = &slice[cached_commits_idx..cached_commits_idx + max_cached * DIGEST_SIZE];
    let cached_commits: &[[F; DIGEST_SIZE]] = unsafe {
        std::slice::from_raw_parts(
            cached_commits.as_ptr() as *const [F; DIGEST_SIZE],
            max_cached,
        )
    };

    ProofShapeVarCols {
        idx_flags: &slice[flags_idx..part_cached_idx],
        part_cached_mult: &slice[part_cached_idx..cached_commits_idx],
        cached_commits,
    }
}

fn borrow_var_cols_mut<F>(
    slice: &mut [F],
    idx_flags: usize,
    max_cached: usize,
) -> ProofShapeVarColsMut<'_, F> {
    let flags_idx = 0;
    let part_cached_idx = flags_idx + idx_flags;
    let cached_commits_idx = part_cached_idx + max_cached;

    let cached_commits =
        &mut slice[cached_commits_idx..cached_commits_idx + max_cached * DIGEST_SIZE];
    let cached_commits: &mut [[F; DIGEST_SIZE]] = unsafe {
        std::slice::from_raw_parts_mut(cached_commits.as_ptr() as *mut [F; DIGEST_SIZE], max_cached)
    };

    let (idx_flags, part_cached_mult) =
        slice[flags_idx..cached_commits_idx].split_at_mut(part_cached_idx);

    ProofShapeVarColsMut {
        idx_flags,
        part_cached_mult,
        cached_commits,
    }
}
