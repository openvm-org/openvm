use core::cmp::Reverse;
use std::sync::Arc;

use itertools::{izip, Itertools};
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::{
    keygen::types::{MultiStarkVerifyingKey, VerifierSinglePreprocessedData},
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    AirRef, FiatShamirTranscript, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    primitives::{
        bus::{PowerCheckerBus, RangeCheckerBus},
        pow::PowerCheckerCpuTraceGenerator,
        range::{RangeCheckerAir, RangeCheckerCpuTraceGenerator},
    },
    proof_shape::{
        bus::{NumPublicValuesBus, ProofShapePermutationBus, StartingTidxBus},
        proof_shape::ProofShapeAir,
        pvs::PublicValuesAir,
    },
    system::{
        frame::MultiStarkVkeyFrame, AirModule, BusIndexManager, BusInventory, GlobalCtxCpu,
        Preflight, ProofShapePreflight, TraceGenModule, POW_CHECKER_HEIGHT,
    },
    tracegen::{ModuleChip, RowMajorChip},
};

pub mod bus;
#[allow(clippy::module_inception)]
pub mod proof_shape;
pub mod pvs;

#[cfg(feature = "cuda")]
mod cuda_abi;

#[derive(Clone)]
pub struct AirMetadata {
    is_required: bool,
    num_public_values: usize,
    num_interactions: usize,
    main_width: usize,
    cached_widths: Vec<usize>,
    preprocessed_width: Option<usize>,
    preprocessed_data: Option<VerifierSinglePreprocessedData<Digest>>,
}

pub struct ProofShapeModule {
    // Verifying key fields
    per_air: Vec<AirMetadata>,
    l_skip: usize,
    /// Threshold from the child VK used by [`ProofShapeAir`] on the summary row:
    /// `sum_i(num_interactions[i] * lifted_height[i]) < max_interaction_count`,
    /// with `lifted_height[i] = max(trace_height[i], 2^l_skip)`.
    max_interaction_count: u32,

    // Buses (inventory for external, others are internal)
    bus_inventory: BusInventory,
    range_bus: RangeCheckerBus,
    pow_bus: PowerCheckerBus,
    permutation_bus: ProofShapePermutationBus,
    starting_tidx_bus: StartingTidxBus,
    num_pvs_bus: NumPublicValuesBus,

    // Required for ProofShapeAir tracegen + constraints
    idx_encoder: Arc<Encoder>,
    min_cached_idx: usize,
    max_cached: usize,
    commit_mult: usize,

    // Module sends extra public values message for use outside of verifier
    // sub-circuit if true
    continuations_enabled: bool,
}

impl ProofShapeModule {
    pub fn new(
        mvk: &MultiStarkVkeyFrame,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        continuations_enabled: bool,
    ) -> Self {
        let idx_encoder = Arc::new(Encoder::new(mvk.per_air.len(), 2, true));

        let (min_cached_idx, min_cached) = mvk
            .per_air
            .iter()
            .enumerate()
            .min_by_key(|(_, avk)| avk.params.width.cached_mains.len())
            .map(|(idx, avk)| (idx, avk.params.width.cached_mains.len()))
            .unwrap();
        let mut max_cached = mvk
            .per_air
            .iter()
            .map(|avk| avk.params.width.cached_mains.len())
            .max()
            .unwrap();
        if min_cached == max_cached {
            max_cached += 1;
        }

        let per_air = mvk
            .per_air
            .iter()
            .map(|avk| AirMetadata {
                is_required: avk.is_required,
                num_public_values: avk.params.num_public_values,
                num_interactions: avk.num_interactions,
                main_width: avk.params.width.common_main,
                cached_widths: avk.params.width.cached_mains.clone(),
                preprocessed_width: avk.params.width.preprocessed,
                preprocessed_data: avk.preprocessed_data.clone(),
            })
            .collect_vec();

        let range_bus = bus_inventory.range_checker_bus;
        let pow_bus = bus_inventory.power_checker_bus;
        Self {
            per_air,
            l_skip: mvk.params.l_skip,
            max_interaction_count: mvk.params.logup.max_interaction_count,
            bus_inventory,
            range_bus,
            pow_bus,
            permutation_bus: ProofShapePermutationBus::new(b.new_bus_idx()),
            starting_tidx_bus: StartingTidxBus::new(b.new_bus_idx()),
            num_pvs_bus: NumPublicValuesBus::new(b.new_bus_idx()),
            idx_encoder,
            min_cached_idx,
            max_cached,
            commit_mult: mvk.params.whir.rounds.first().unwrap().num_queries,
            continuations_enabled,
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
    {
        let l_skip = child_vk.inner.params.l_skip;
        ts.observe_commit(child_vk.pre_hash);
        ts.observe_commit(proof.common_main_commit);

        let mut pvs_tidx = vec![];
        let mut starting_tidx = vec![];

        for (trace_vdata, avk, pvs) in izip!(
            &proof.trace_vdata,
            &child_vk.inner.per_air,
            &proof.public_values
        ) {
            let is_air_present = trace_vdata.is_some();
            starting_tidx.push(ts.len());

            if !avk.is_required {
                ts.observe(F::from_bool(is_air_present));
            }
            if let Some(trace_vdata) = trace_vdata {
                if let Some(pdata) = avk.preprocessed_data.as_ref() {
                    ts.observe_commit(pdata.commit);
                } else {
                    ts.observe(F::from_usize(trace_vdata.log_height));
                }
                debug_assert_eq!(avk.num_cached_mains(), trace_vdata.cached_commitments.len());
                if !pvs.is_empty() {
                    pvs_tidx.push(ts.len());
                }
                for commit in &trace_vdata.cached_commitments {
                    ts.observe_commit(*commit);
                }
                debug_assert_eq!(avk.params.num_public_values, pvs.len());
            }
            for pv in pvs {
                ts.observe(*pv);
            }
        }

        let mut sorted_trace_vdata: Vec<_> = proof
            .trace_vdata
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(air_id, data)| data.map(|data| (air_id, data)))
            .collect();
        sorted_trace_vdata.sort_by_key(|(air_idx, data)| (Reverse(data.log_height), *air_idx));

        let n_max = proof
            .trace_vdata
            .iter()
            .flat_map(|datum| {
                datum
                    .as_ref()
                    .map(|datum| datum.log_height.saturating_sub(l_skip))
            })
            .max()
            .unwrap();
        let num_layers = proof.gkr_proof.claims_per_layer.len();
        let n_logup = num_layers.saturating_sub(l_skip);

        preflight.proof_shape = ProofShapePreflight {
            sorted_trace_vdata,
            starting_tidx,
            pvs_tidx,
            post_tidx: ts.len(),
            n_max,
            n_logup,
            l_skip: child_vk.inner.params.l_skip,
        };
    }
}

impl AirModule for ProofShapeModule {
    fn num_airs(&self) -> usize {
        3
    }

    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let proof_shape_air = ProofShapeAir::<4, 8> {
            per_air: self.per_air.clone(),
            l_skip: self.l_skip,
            min_cached_idx: self.min_cached_idx,
            max_cached: self.max_cached,
            commit_mult: self.commit_mult,
            max_interaction_count: self.max_interaction_count,
            idx_encoder: self.idx_encoder.clone(),
            range_bus: self.range_bus,
            pow_bus: self.pow_bus,
            permutation_bus: self.permutation_bus,
            starting_tidx_bus: self.starting_tidx_bus,
            num_pvs_bus: self.num_pvs_bus,
            fraction_folder_input_bus: self.bus_inventory.fraction_folder_input_bus,
            expression_claim_n_max_bus: self.bus_inventory.expression_claim_n_max_bus,
            gkr_module_bus: self.bus_inventory.gkr_module_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            hyperdim_bus: self.bus_inventory.hyperdim_bus,
            lifted_heights_bus: self.bus_inventory.lifted_heights_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            n_lift_bus: self.bus_inventory.n_lift_bus,
            cached_commit_bus: self.bus_inventory.cached_commit_bus,
            continuations_enabled: self.continuations_enabled,
        };
        let pvs_air = PublicValuesAir {
            public_values_bus: self.bus_inventory.public_values_bus,
            num_pvs_bus: self.num_pvs_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            continuations_enabled: self.continuations_enabled,
        };
        let range_checker = RangeCheckerAir::<8> {
            bus: self.range_bus,
        };
        vec![
            Arc::new(proof_shape_air) as AirRef<_>,
            Arc::new(pvs_air) as AirRef<_>,
            Arc::new(range_checker) as AirRef<_>,
        ]
    }
}

impl TraceGenModule<GlobalCtxCpu, CpuBackend<BabyBearPoseidon2Config>> for ProofShapeModule {
    type ModuleSpecificCtx<'a> = Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>;

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        pow_checker: &Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>> {
        let range_checker = Arc::new(RangeCheckerCpuTraceGenerator::<8>::default());
        let proof_shape = proof_shape::ProofShapeChip::<4, 8>::new(
            self.idx_encoder.clone(),
            self.min_cached_idx,
            self.max_cached,
            range_checker.clone(),
            pow_checker.clone(),
        );
        let ctx = (child_vk, proofs, preflights);
        let chips = [
            ProofShapeModuleChip::ProofShape(proof_shape),
            ProofShapeModuleChip::PublicValues,
        ];
        let mut ctxs: Vec<_> = chips
            .par_iter()
            .map(|chip| {
                chip.generate_proving_ctx(
                    &ctx,
                    required_heights.map(|heights| heights[chip.index()]),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Option<Vec<_>>>()?;

        tracing::trace_span!("wrapper.generate_trace", air = "RangeChecker").in_scope(|| {
            ctxs.push(AirProvingContext::simple_no_pis(
                ColMajorMatrix::from_row_major(&range_checker.generate_trace_row_major()),
            ));
        });
        Some(ctxs)
    }
}

#[derive(strum_macros::Display, strum::EnumDiscriminants)]
#[strum_discriminants(repr(usize))]
enum ProofShapeModuleChip {
    ProofShape(proof_shape::ProofShapeChip<4, 8>),
    PublicValues,
}

impl ProofShapeModuleChip {
    fn index(&self) -> usize {
        ProofShapeModuleChipDiscriminants::from(self) as usize
    }
}

impl RowMajorChip<F> for ProofShapeModuleChip {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a [Proof<BabyBearPoseidon2Config>],
        &'a [Preflight],
    );

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        use ProofShapeModuleChip::*;
        match self {
            ProofShape(chip) => chip.generate_trace(ctx, required_height),
            PublicValues => {
                pvs::PublicValuesTraceGenerator.generate_trace(&(ctx.1, ctx.2), required_height)
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use openvm_cuda_backend::GpuBackend;

    use super::*;
    use crate::{
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu, GlobalCtxGpu},
        primitives::{
            pow::cuda::PowerCheckerGpuTraceGenerator, range::cuda::RangeCheckerGpuTraceGenerator,
        },
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackend> for ProofShapeModule {
        type ModuleSpecificCtx<'a> = Arc<PowerCheckerGpuTraceGenerator<2, POW_CHECKER_HEIGHT>>;

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            pow_checker_gpu: &Arc<PowerCheckerGpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            use crate::tracegen::ModuleChip;

            let range_checker_gpu = Arc::new(RangeCheckerGpuTraceGenerator::<8>::default());
            let proof_shape_chip = proof_shape::cuda::ProofShapeChipGpu::<4, 8>::new(
                self.idx_encoder.width(),
                self.min_cached_idx,
                self.max_cached,
                range_checker_gpu.clone(),
                pow_checker_gpu.clone(),
            );
            let mut ctxs = Vec::with_capacity(3);
            // PERF[jpw]: we avoid par_iter so that kernel launches occur on the same stream.
            // This can be parallelized to separate streams for more CUDA stream parallelism, but it
            // will require recording events so streams properly sync for cudaMemcpyAsync and kernel
            // launches
            let proof_shape_ctx =
                tracing::trace_span!("wrapper.generate_trace", air = "ProofShape").in_scope(
                    || {
                        proof_shape_chip.generate_proving_ctx(
                            &(child_vk, preflights),
                            required_heights.map(|heights| heights[0]),
                        )
                    },
                )?;
            ctxs.push(proof_shape_ctx);

            let public_values_ctx =
                tracing::trace_span!("wrapper.generate_trace", air = "PublicValues").in_scope(
                    || {
                        pvs::cuda::PublicValuesGpuTraceGenerator.generate_proving_ctx(
                            &(proofs, preflights),
                            required_heights.map(|heights| heights[1]),
                        )
                    },
                )?;
            ctxs.push(public_values_ctx);
            // Drop the proof_shape chip so we can finalize auxiliary trace state (it holds Arc
            // clones).
            drop(proof_shape_chip);
            // Caution: proof_shape **must** finish trace gen before we materialize range checker
            // trace or sync power checker multiplicities to CPU.
            tracing::trace_span!("wrapper.generate_trace", air = "RangeChecker").in_scope(|| {
                ctxs.push(AirProvingContext::simple_no_pis(
                    Arc::try_unwrap(range_checker_gpu).unwrap().generate_trace(),
                ));
            });

            Some(ctxs)
        }
    }
}
