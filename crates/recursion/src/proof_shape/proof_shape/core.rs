use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use itertools::fold;
use openvm_circuit_primitives::{
    encoder::Encoder,
    utils::{and, not, or, select},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, keygen::types::MultiStarkVerifyingKey, proof::Proof,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        AirShapeBus, AirShapeBusMessage, AirShapeProperty, CachedCommitBus, CachedCommitBusMessage,
        CommitmentsBus, CommitmentsBusMessage, ExpressionClaimNMaxBus, ExpressionClaimNMaxMessage,
        FractionFolderInputBus, FractionFolderInputMessage, GkrModuleBus, GkrModuleMessage,
        HyperdimBus, HyperdimBusMessage, LiftedHeightsBus, LiftedHeightsBusMessage, TranscriptBus,
        TranscriptBusMessage,
    },
    primitives::{
        bus::{PowerCheckerBus, PowerCheckerBusMessage, RangeCheckerBus, RangeCheckerBusMessage},
        pow::PowerCheckerTraceGenerator,
        range::RangeCheckerTraceGenerator,
    },
    proof_shape::{
        bus::{
            NumPublicValuesBus, NumPublicValuesMessage, ProofShapePermutationBus,
            ProofShapePermutationMessage, StartingTidxBus, StartingTidxMessage,
        },
        AirMetadata,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    tracegen::RowMajorChip,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct ProofShapeCols<F, const NUM_LIMBS: usize> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    pub idx: F,
    pub sorted_idx: F,
    /// Represents log2 trace height when `is_present`.
    ///
    /// Has a special use on summary row (when `is_last`).
    pub log_height: F,
    /// When `is_present`, constrained to equal `log_height - l_skip < 0 ? 1 : 0`.
    pub n_sign_bit: F,
    /// Whether this AIR needs rotation openings.
    pub need_rot: F,

    // First possible tidx and non-main cidx of the current AIR
    pub starting_tidx: F,
    pub starting_cidx: F,

    // Columns that may be read from the transcript. Note that cached_commits is also read
    // from the transcript.
    pub is_present: F,

    /// Will be constrained to be `2^log_height` when `is_present`.
    ///
    /// Has a special use on summary row (when `is_last`).
    pub height: F,

    // Number of present AIRs so far
    pub num_present: F,

    // The total number of interactions over all traces needs to fit in a single field element,
    // so we assume that it only requires INTERACTIONS_LIMBS (4) limbs to store.
    //
    // To constrain the correctness of n_logup, we ensure that `total_interactions_limbs` has
    // _exactly_ `CELLS_LIMBS * LIMB_BITS - (l_skip + n_logup)` leading zeroes. We do this by
    // a) recording the most significant non-zero limb i and b) making sure
    // total_interaction_limbs[i] * 2^{the number of remaining leading zeroes} is within [0,
    // 256).
    //
    // To constrain that the total number of interactions over all traces is less than the
    // max interactions set in the vk, we record the most significant limb at which the max
    // limb decomposition and total_interactions_limbs differ. The difference between those
    // two limbs is then range checked to be within [1, 256).
    pub lifted_height_limbs: [F; NUM_LIMBS],
    pub num_interactions_limbs: [F; NUM_LIMBS],
    pub total_interactions_limbs: [F; NUM_LIMBS],

    /// The maximum hypercube dimension across all present AIR traces, or zero.
    /// Computed as max(0, n0, n1, ...) where ni = log_height_i - l_skip for each present trace.
    pub n_max: F,
    pub is_n_max_greater: F,

    pub num_air_id_lookups: F,
    pub num_columns: F,
}

// Variable-length columns are stored at the end
pub struct ProofShapeVarCols<'a, F> {
    pub idx_flags: &'a [F],                     // [F; IDX_FLAGS]
    pub cached_commits: &'a [[F; DIGEST_SIZE]], // [[F; DIGEST_SIZE]; MAX_CACHED]
}

pub struct ProofShapeVarColsMut<'a, F> {
    pub idx_flags: &'a mut [F],                     // [F; IDX_FLAGS]
    pub cached_commits: &'a mut [[F; DIGEST_SIZE]], // [[F; DIGEST_SIZE]; MAX_CACHED]
}

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
                    .fold(0, |acc, interaction| acc + interaction.message.len() + 2)
        })
        .collect::<Vec<_>>()
}

#[derive(derive_new::new)]
pub(in crate::proof_shape) struct ProofShapeChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    idx_encoder: Arc<Encoder>,
    min_cached_idx: usize,
    max_cached: usize,
    range_checker: Arc<RangeCheckerTraceGenerator<LIMB_BITS>>,
    pow_checker: Arc<PowerCheckerTraceGenerator<2, 32>>,
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

                cols.idx = F::from_usize(*idx);
                cols.sorted_idx = F::from_usize(sorted_idx);
                cols.log_height = F::from_usize(log_height);
                cols.n_sign_bit = F::from_bool(n.is_negative());
                cols.need_rot = F::from_bool(child_vk.inner.per_air[*idx].params.need_rot);
                sorted_idx += 1;

                cols.starting_tidx = F::from_usize(preflight.proof_shape.starting_tidx[*idx]);
                cols.starting_cidx = F::from_usize(cidx);
                let has_preprocessed = child_vk.inner.per_air[*idx].preprocessed_data.is_some();
                cidx += has_preprocessed as usize;

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
                let trace_width = &child_vk.inner.per_air[*idx].params.width;
                let num_columns = trace_width.common_main
                    + trace_width.preprocessed.iter().copied().sum::<usize>()
                    + trace_width.cached_mains.iter().copied().sum::<usize>();
                cols.num_columns = F::from_usize(num_columns);

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

                cols.idx = F::from_usize(idx);
                cols.sorted_idx = F::from_usize(sorted_idx);
                sorted_idx += 1;
                cols.need_rot = F::ZERO;

                cols.num_present = num_present;

                cols.starting_tidx = F::from_usize(preflight.proof_shape.starting_tidx[idx]);
                cols.starting_cidx = F::from_usize(cidx);

                cols.total_interactions_limbs = total_interactions_f;
                cols.n_max = F::from_usize(preflight.proof_shape.n_max);
                cols.num_columns = F::ZERO;

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
                cols.need_rot = F::ZERO;
                cols.num_columns = F::ZERO;
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
                // msb_limb_zero_bits_exp
                cols.log_height = F::from_usize(1 << msb_limb_zero_bits);

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
                cols.n_max = F::from_usize(preflight.proof_shape.n_max);
                cols.is_n_max_greater = F::from_bool(preflight.proof_shape.n_max > n_logup);

                // n_logup
                cols.starting_cidx = F::from_usize(n_logup);

                range_checker
                    .add_count(msb_limb.as_canonical_u32() as usize * (1 << msb_limb_zero_bits));
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
    pub starting_tidx_bus: StartingTidxBus,
    pub num_pvs_bus: NumPublicValuesBus,

    // Inter-module buses
    pub gkr_module_bus: GkrModuleBus,
    pub air_shape_bus: AirShapeBus,
    pub expression_claim_n_max_bus: ExpressionClaimNMaxBus,
    pub fraction_folder_input_bus: FractionFolderInputBus,
    pub hyperdim_bus: HyperdimBus,
    pub lifted_heights_bus: LiftedHeightsBus,
    pub commitments_bus: CommitmentsBus,
    pub transcript_bus: TranscriptBus,

    // For continuations
    pub cached_commit_bus: CachedCommitBus,
    pub continuations_enabled: bool,
}

impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ProofShapeAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        ProofShapeCols::<F, NUM_LIMBS>::width()
            + self.idx_encoder.width()
            + self.max_cached * DIGEST_SIZE
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

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
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

        builder
            .when(local.is_first)
            .assert_eq(local.is_present, local.num_present);
        builder.when(local.is_valid).assert_eq(
            local.num_present + next.is_present * next.is_valid,
            next.num_present,
        );

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
        builder
            .when(and(not(local.is_present), local.is_valid))
            .assert_zero(local.log_height);

        // Range check difference using ExponentBus to ensure local.log_height >= next.log_height
        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: local.log_height - next.log_height,
                max_bits: AB::Expr::from_usize(5),
            },
            and(local.is_valid, not(next.is_last)),
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // VK FIELD SELECTION
        ///////////////////////////////////////////////////////////////////////////////////////////
        let mut num_interactions_per_row = [AB::Expr::ZERO; NUM_LIMBS];

        // Select values for TranscriptBus
        let mut is_required = AB::Expr::ZERO;
        let mut is_min_cached = AB::Expr::ZERO;
        let mut has_preprocessed = AB::Expr::ZERO;
        let mut cached_present = vec![AB::Expr::ZERO; self.max_cached];

        // Select values for AirShapeBus
        let mut num_interactions = AB::Expr::ZERO;

        // Select values for LiftedHeightsBus
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

            when_current.assert_eq(local.idx, AB::F::from_usize(i));

            main_common_width += is_current_air.clone() * AB::F::from_usize(air_data.main_width);

            if air_data.num_public_values != 0 {
                has_pvs += is_current_air.clone();
            }
            num_pvs += is_current_air.clone() * AB::F::from_usize(air_data.num_public_values);

            // Select number of interactions for use later in the AIR and constrain that the
            // num_interactions_per_row limb decomposition is correct.
            num_interactions +=
                is_current_air.clone() * AB::F::from_usize(air_data.num_interactions);

            for (i, &limb) in decompose_f::<AB::F, NUM_LIMBS, LIMB_BITS>(air_data.num_interactions)
                .iter()
                .enumerate()
            {
                num_interactions_per_row[i] += is_current_air.clone() * limb;
            }

            if air_data.is_required {
                is_required += is_current_air.clone();
                when_current.assert_one(local.is_present);
            }

            if i == self.min_cached_idx {
                is_min_cached += is_current_air.clone();
            }

            if let Some(preprocessed) = &air_data.preprocessed_data {
                when_current.assert_eq(
                    local.log_height,
                    AB::Expr::from_usize(
                        self.l_skip.wrapping_add_signed(preprocessed.hypercube_dim),
                    ),
                );
                has_preprocessed += is_current_air.clone();

                preprocessed_stacked_width += is_current_air.clone()
                    * AB::F::from_usize(air_data.preprocessed_width.unwrap());
                (0..DIGEST_SIZE).for_each(|didx| {
                    preprocessed_commit[didx] += is_current_air.clone()
                        * AB::F::from_u32(preprocessed.commit[didx].as_canonical_u32());
                });
            }

            for (cached_idx, width) in air_data.cached_widths.iter().enumerate() {
                cached_present[cached_idx] += is_current_air.clone();
                cached_widths[cached_idx] += is_current_air.clone() * AB::Expr::from_usize(*width);
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // TRANSCRIPT OBSERVATIONS
        ///////////////////////////////////////////////////////////////////////////////////////////
        let is_first_idx = self.idx_encoder.get_flag_expr::<AB>(0, localv.idx_flags);
        builder
            .when(is_first_idx.clone())
            .assert_eq(local.starting_tidx, AB::Expr::from_usize(2 * DIGEST_SIZE));

        self.starting_tidx_bus.receive(
            builder,
            local.proof_idx,
            StartingTidxMessage {
                air_idx: local.idx * local.is_valid
                    + AB::Expr::from_usize(self.per_air.len()) * local.is_last,
                tidx: local.starting_tidx.into(),
            },
            or(
                local.is_last,
                and(local.is_valid, not::<AB::Expr>(is_first_idx)),
            ),
        );

        let mut tidx = local.starting_tidx.into();
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
        tidx += not::<AB::Expr>(is_required) * local.is_valid;

        for (didx, commit_val) in preprocessed_commit.iter().enumerate() {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_usize(didx),
                    value: commit_val.clone(),
                    is_sample: AB::Expr::ZERO,
                },
                has_preprocessed.clone() * local.is_present,
            );
        }
        tidx += has_preprocessed.clone() * AB::Expr::from_usize(DIGEST_SIZE) * local.is_present;

        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: tidx.clone(),
                value: local.log_height.into(),
                is_sample: AB::Expr::ZERO,
            },
            not::<AB::Expr>(has_preprocessed.clone()) * local.is_present,
        );
        tidx += not::<AB::Expr>(has_preprocessed.clone()) * local.is_present;

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
                    cached_present[i].clone() * local.is_present,
                );
                tidx += cached_present[i].clone() * local.is_present;
            }
        });

        let num_pvs_tidx = tidx.clone();
        tidx += num_pvs.clone() * local.is_present;

        self.starting_tidx_bus.send(
            builder,
            local.proof_idx,
            StartingTidxMessage {
                air_idx: local.idx + AB::F::ONE,
                tidx,
            },
            local.is_valid,
        );

        for didx in 0..DIGEST_SIZE {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(didx),
                    value: localv.cached_commits[self.max_cached - 1][didx].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_last,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_usize(didx + DIGEST_SIZE),
                    value: localv.cached_commits[self.max_cached - 1][didx].into(),
                    is_sample: AB::Expr::ZERO,
                },
                is_min_cached.clone() * local.is_valid,
            );
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // AIR SHAPE LOOKUP
        ///////////////////////////////////////////////////////////////////////////////////////////
        self.air_shape_bus.send(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sorted_idx.into(),
                property_idx: AirShapeProperty::AirId.to_field(),
                value: local.idx.into(),
            },
            local.is_present * local.num_air_id_lookups,
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

        self.air_shape_bus.send(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sorted_idx.into(),
                property_idx: AirShapeProperty::NeedRot.to_field(),
                value: local.need_rot.into(),
            },
            local.is_present * local.num_columns,
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // HYPERDIM (SIGNED N) LOOKUP
        ///////////////////////////////////////////////////////////////////////////////////////////
        let l_skip = AB::F::from_usize(self.l_skip);
        let n = local.log_height.into() - l_skip;
        builder.assert_bool(local.n_sign_bit);
        builder.assert_bool(local.need_rot);
        builder
            .when(not(local.is_present))
            .assert_zero(local.need_rot);
        builder
            .when(not(local.is_present))
            .assert_zero(local.num_columns);
        let n_abs = select(local.n_sign_bit, -n.clone(), n.clone());
        // We range check `n_abs` is in `[0, 32)`.
        // We constrain `n = n_sign_bit ? -n_abs : n_abs` and `n := log_height - l_skip`.
        // This implies `log_height - l_skip` is in `(-32, 32)` and `n_abs` is its absolute value.
        // We further use PowerCheckerBus below to range check that `log_height` is in `[0, 32)`.
        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: n_abs.clone(),
                max_bits: AB::Expr::from_usize(5),
            },
            local.is_present,
        );

        self.hyperdim_bus.send(
            builder,
            local.proof_idx,
            HyperdimBusMessage {
                sort_idx: local.sorted_idx.into(),
                n_abs: n_abs.clone(),
                n_sign_bit: local.n_sign_bit.into(),
            },
            local.is_present * (local.num_air_id_lookups + AB::F::ONE),
        );

        ///////////////////////////////////////////////////////////////////////////////////////////
        // LIFTED HEIGHTS LOOKUP + STACKING COMMITMENTS
        ///////////////////////////////////////////////////////////////////////////////////////////
        // lifted_height = max(2^log_height, 2^l_skip)
        let lifted_height = select(
            local.n_sign_bit,
            AB::F::from_usize(1 << self.l_skip),
            local.height,
        );
        let log_lifted_height = not(local.n_sign_bit) * n_abs.clone() + l_skip;

        self.pow_bus.lookup_key(
            builder,
            PowerCheckerBusMessage {
                log: local.log_height.into(),
                exp: local.height.into(),
            },
            local.is_present,
        );

        self.lifted_heights_bus.send(
            builder,
            local.proof_idx,
            LiftedHeightsBusMessage {
                sort_idx: local.sorted_idx.into(),
                part_idx: AB::Expr::ZERO,
                commit_idx: AB::Expr::ZERO,
                hypercube_dim: n.clone(),
                lifted_height: lifted_height.clone(),
                log_lifted_height: log_lifted_height.clone(),
            },
            local.is_present * main_common_width,
        );

        builder
            .when(and(local.is_first, local.is_valid))
            .assert_one(local.starting_cidx);
        let mut cidx_offset = AB::Expr::ZERO;

        self.lifted_heights_bus.send(
            builder,
            local.proof_idx,
            LiftedHeightsBusMessage {
                sort_idx: local.sorted_idx.into(),
                part_idx: cidx_offset.clone() + AB::F::ONE,
                commit_idx: cidx_offset.clone() + local.starting_cidx,
                hypercube_dim: n.clone(),
                lifted_height: lifted_height.clone(),
                log_lifted_height: log_lifted_height.clone(),
            },
            local.is_present * preprocessed_stacked_width,
        );

        self.commitments_bus.send(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: AB::Expr::ZERO,
                minor_idx: cidx_offset.clone() + local.starting_cidx,
                commitment: preprocessed_commit,
            },
            has_preprocessed.clone() * local.is_valid * AB::Expr::from_usize(self.commit_mult),
        );
        cidx_offset += has_preprocessed.clone();

        (0..self.max_cached).for_each(|cached_idx| {
            self.lifted_heights_bus.send(
                builder,
                local.proof_idx,
                LiftedHeightsBusMessage {
                    sort_idx: local.sorted_idx.into(),
                    part_idx: cidx_offset.clone() + AB::F::ONE,
                    commit_idx: cidx_offset.clone() + local.starting_cidx,
                    hypercube_dim: n.clone(),
                    lifted_height: lifted_height.clone(),
                    log_lifted_height: log_lifted_height.clone(),
                },
                local.is_present * cached_widths[cached_idx].clone(),
            );

            self.commitments_bus.send(
                builder,
                local.proof_idx,
                CommitmentsBusMessage {
                    major_idx: AB::Expr::ZERO,
                    minor_idx: cidx_offset.clone() + local.starting_cidx,
                    commitment: localv.cached_commits[cached_idx].map(Into::into),
                },
                cached_present[cached_idx].clone()
                    * local.is_valid
                    * AB::Expr::from_usize(self.commit_mult),
            );
            cidx_offset += cached_present[cached_idx].clone();

            self.cached_commit_bus.send(
                builder,
                local.proof_idx,
                CachedCommitBusMessage {
                    air_idx: local.idx.into(),
                    cached_idx: AB::Expr::from_usize(cached_idx),
                    cached_commit: localv.cached_commits[cached_idx].map(Into::into),
                },
                cached_present[cached_idx].clone()
                    * local.is_valid
                    * AB::Expr::from_bool(self.continuations_enabled),
            );
        });

        builder
            .when(and(local.is_valid, not(next.is_last)))
            .assert_eq(local.starting_cidx + cidx_offset, next.starting_cidx);

        self.commitments_bus.send(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: AB::Expr::ZERO,
                minor_idx: AB::Expr::ZERO,
                commitment: localv.cached_commits[self.max_cached - 1].map(Into::into),
            },
            is_min_cached.clone() * local.is_valid * AB::Expr::from_usize(self.commit_mult),
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
                local.lifted_height_limbs.iter().enumerate(),
                AB::Expr::ZERO,
                |acc, (i, limb)| acc + (AB::Expr::from_u32(1 << (i * LIMB_BITS)) * *limb),
            ),
            lifted_height,
        );

        for i in 0..NUM_LIMBS {
            self.range_bus.lookup_key(
                builder,
                RangeCheckerBusMessage {
                    value: local.lifted_height_limbs[i].into(),
                    max_bits: AB::Expr::from_usize(LIMB_BITS),
                },
                local.is_valid,
            );
        }

        // Constrain that num_interactions = height * num_interactions_per_row
        let mut carry = vec![AB::Expr::ZERO; NUM_LIMBS * 2];
        let carry_divide = AB::F::from_u32(1 << LIMB_BITS).inverse();

        for (i, &height_limb) in local.lifted_height_limbs.iter().enumerate() {
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
                        max_bits: AB::Expr::from_usize(LIMB_BITS),
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
                    max_bits: AB::Expr::from_usize(LIMB_BITS),
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
            .when(local.sorted_idx - AB::F::from_usize(self.per_air.len() - 1))
            .assert_zero(next.is_last);
        builder
            .when(next.is_last)
            .assert_zero(local.sorted_idx - AB::F::from_usize(self.per_air.len() - 1));

        // Constrain that n_logup is correct, i.e. that there are CELLS_LIMBS * LIMB_BITS - n_logup
        // leading zeroes in total_interactions_limbs. Because we only do this on the is_last row,
        // we can reuse several of our columns to save space.
        //
        // We mark the most significant non-zero limb of local.total_interactions_limbs using the
        // non_zero_marker column array defined below, and the remaining number of leading 0 bits
        // needed within the limb using msb_limb_zero_bits_exp. Column limb_to_range_check is used
        // to store the value of the most significant limb to range check.
        let non_zero_marker = local.lifted_height_limbs;
        let limb_to_range_check = local.height;
        let msb_limb_zero_bits_exp = local.log_height;
        let n_logup = local.starting_cidx;

        let mut prefix = AB::Expr::ZERO;
        let mut expected_limb_to_range_check = AB::Expr::ZERO;
        let mut msb_limb_zero_bits = AB::Expr::ZERO;

        for i in (0..NUM_LIMBS).rev() {
            prefix += non_zero_marker[i].into();
            expected_limb_to_range_check += local.total_interactions_limbs[i] * non_zero_marker[i];
            msb_limb_zero_bits += non_zero_marker[i] * AB::F::from_usize((i + 1) * LIMB_BITS);

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
        msb_limb_zero_bits -= n_logup + prefix * AB::F::from_usize(self.l_skip);

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
                max_bits: AB::Expr::from_usize(LIMB_BITS),
            },
            local.is_last,
        );

        // Constrain n_max on each row. Also constrain that local.is_n_max_greater is one when
        // n_max is greater than n_logup, and zero otherwise.
        builder
            .when(local.is_first)
            .assert_eq(local.n_max, not(local.n_sign_bit) * n_abs);
        builder
            .when(local.is_first)
            .when(local.n_sign_bit)
            .assert_zero(local.n_max);
        builder
            .when(local.is_valid)
            .assert_eq(local.n_max, next.n_max);

        builder.assert_bool(local.is_n_max_greater);
        self.range_bus.lookup_key(
            builder,
            RangeCheckerBusMessage {
                value: (local.n_max - n_logup) * (local.is_n_max_greater * AB::F::TWO - AB::F::ONE),
                max_bits: AB::Expr::from_usize(5),
            },
            local.is_last,
        );

        self.gkr_module_bus.send(
            builder,
            local.proof_idx,
            GkrModuleMessage {
                tidx: local.starting_tidx.into(),
                n_logup: n_logup.into(),
                n_max: local.n_max.into(),
                is_n_max_greater: local.is_n_max_greater.into(),
            },
            local.is_last,
        );

        // Send n_max value to expression claim air
        self.expression_claim_n_max_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimNMaxMessage {
                n_max: local.n_max.into(),
            },
            local.is_last,
        );

        // Send count of present airs to fraction folder air
        self.fraction_folder_input_bus.send(
            builder,
            local.proof_idx,
            FractionFolderInputMessage {
                num_present_airs: local.num_present,
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
                max_bits: AB::Expr::from_usize(LIMB_BITS),
            },
            local.is_last,
        );
    }
}

fn decompose_f<F: PrimeCharacteristicRing, const LIMBS: usize, const LIMB_BITS: usize>(
    value: usize,
) -> [F; LIMBS] {
    from_fn(|i| F::from_usize((value >> (i * LIMB_BITS)) & ((1 << LIMB_BITS) - 1)))
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
    let cached_commits_idx = flags_idx + idx_flags;

    let cached_commits = &slice[cached_commits_idx..cached_commits_idx + max_cached * DIGEST_SIZE];
    let cached_commits: &[[F; DIGEST_SIZE]] = unsafe {
        std::slice::from_raw_parts(
            cached_commits.as_ptr() as *const [F; DIGEST_SIZE],
            max_cached,
        )
    };

    ProofShapeVarCols {
        idx_flags: &slice[flags_idx..cached_commits_idx],
        cached_commits,
    }
}

fn borrow_var_cols_mut<F>(
    slice: &mut [F],
    idx_flags: usize,
    max_cached: usize,
) -> ProofShapeVarColsMut<'_, F> {
    let flags_idx = 0;
    let cached_commits_idx = flags_idx + idx_flags;

    let cached_commits =
        &mut slice[cached_commits_idx..cached_commits_idx + max_cached * DIGEST_SIZE];
    let cached_commits: &mut [[F; DIGEST_SIZE]] = unsafe {
        std::slice::from_raw_parts_mut(cached_commits.as_ptr() as *mut [F; DIGEST_SIZE], max_cached)
    };

    ProofShapeVarColsMut {
        idx_flags: &mut slice[flags_idx..cached_commits_idx],
        cached_commits,
    }
}
