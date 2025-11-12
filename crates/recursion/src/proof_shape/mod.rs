use core::cmp::Reverse;
use std::sync::Arc;

use itertools::{Itertools, izip};
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    Digest, F,
    keygen::types::{MultiStarkVerifyingKeyV2, VerifierSinglePreprocessedData},
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::Proof,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
};

use crate::{
    primitives::{
        bus::{PowerCheckerBus, RangeCheckerBus},
        pow::PowerCheckerTraceGenerator,
        range::{RangeCheckerAir, RangeCheckerTraceGenerator},
    },
    proof_shape::{
        bus::{NumPublicValuesBus, ProofShapePermutationBus, StartingTidxBus},
        proof_shape::ProofShapeAir,
        pvs::PublicValuesAir,
    },
    system::{
        AirModule, BusIndexManager, BusInventory, GlobalCtxCpu, ModuleChip, Preflight,
        ProofShapePreflight, TraceGenModule, frame::MultiStarkVkeyFrame,
    },
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
    range_checker: Arc<RangeCheckerTraceGenerator<8>>,
    pow_checker: Arc<PowerCheckerTraceGenerator<2, 32>>,
}

impl ProofShapeModule {
    pub fn new(
        mvk: &MultiStarkVkeyFrame,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        pow_checker: Arc<PowerCheckerTraceGenerator<2, 32>>,
    ) -> Self {
        let idx_encoder = Arc::new(Encoder::new(mvk.per_air.len(), 2, true));
        let range_checker = Arc::new(RangeCheckerTraceGenerator::<8>::default());

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
            bus_inventory,
            range_bus,
            pow_bus,
            permutation_bus: ProofShapePermutationBus::new(b.new_bus_idx()),
            starting_tidx_bus: StartingTidxBus::new(b.new_bus_idx()),
            num_pvs_bus: NumPublicValuesBus::new(b.new_bus_idx()),
            idx_encoder,
            min_cached_idx,
            max_cached,
            commit_mult: mvk.params.num_whir_queries,
            range_checker,
            pow_checker,
        }
    }

    pub fn run_preflight<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) where
        TS: FiatShamirTranscript + TranscriptHistory,
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
                    ts.observe(F::from_canonical_usize(trace_vdata.log_height));
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
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let proof_shape_air = ProofShapeAir::<4, 8> {
            per_air: self.per_air.clone(),
            l_skip: self.l_skip,
            min_cached_idx: self.min_cached_idx,
            max_cached: self.max_cached,
            commit_mult: self.commit_mult,
            idx_encoder: self.idx_encoder.clone(),
            range_bus: self.range_bus,
            pow_bus: self.pow_bus,
            permutation_bus: self.permutation_bus,
            starting_tidx_bus: self.starting_tidx_bus,
            num_pvs_bus: self.num_pvs_bus,
            expression_claim_n_max_bus: self.bus_inventory.expression_claim_n_max_bus,
            gkr_module_bus: self.bus_inventory.gkr_module_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            hyperdim_bus: self.bus_inventory.hyperdim_bus,
            lifted_heights_bus: self.bus_inventory.lifted_heights_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        let pvs_air = PublicValuesAir {
            public_values_bus: self.bus_inventory.public_values_bus,
            num_pvs_bus: self.num_pvs_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
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

/// Empty blob
struct ProofShapeBlob;

impl TraceGenModule<GlobalCtxCpu, CpuBackendV2> for ProofShapeModule {
    type ModuleSpecificCtx = ();

    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        _ctx: (),
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        let proof_shape = proof_shape::ProofShapeChip::<4, 8>::new(
            self.idx_encoder.clone(),
            self.min_cached_idx,
            self.max_cached,
            self.range_checker.clone(),
            self.pow_checker.clone(),
        );
        let blob = ProofShapeBlob;
        let mut ctxs = Vec::with_capacity(3);
        ctxs.extend(
            [
                ProofShapeModuleChip::ProofShape(proof_shape),
                ProofShapeModuleChip::PublicValues,
            ]
            .par_iter()
            .map(|chip| chip.generate_trace(child_vk, proofs, preflights, &blob))
            .collect::<Vec<_>>(),
        );
        ctxs.push(ColMajorMatrix::from_row_major(
            &self.range_checker.generate_trace_row_major(),
        ));
        ctxs.into_iter()
            .map(AirProvingContextV2::simple_no_pis)
            .collect()
    }
}

enum ProofShapeModuleChip {
    ProofShape(proof_shape::ProofShapeChip<4, 8>),
    PublicValues,
}

impl ModuleChip<GlobalCtxCpu, CpuBackendV2> for ProofShapeModuleChip {
    type ModuleSpecificCtx = ProofShapeBlob;

    fn generate_trace(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        _: &ProofShapeBlob,
    ) -> ColMajorMatrix<F> {
        use ProofShapeModuleChip::*;
        let trace = match self {
            ProofShape(chip) => chip.generate_trace(child_vk, proofs, preflights),
            PublicValues => pvs::generate_trace(proofs, preflights),
        };
        ColMajorMatrix::from_row_major(&trace)
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use cuda_backend_v2::GpuBackendV2;
    use openvm_cuda_backend::base::DeviceMatrix;
    use openvm_cuda_common::copy::MemCopyD2H;
    use openvm_stark_backend::prover::MatrixDimensions;
    use p3_field::PrimeField32;

    use super::*;
    use crate::{
        cuda::{GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
        primitives::{
            pow::{PowerCheckerTraceGenerator, cuda::PowerCheckerGpuTraceGenerator},
            range::cuda::RangeCheckerGpuTraceGenerator,
        },
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackendV2> for ProofShapeModule {
        type ModuleSpecificCtx = ();

        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            _ctx: (),
        ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
            let range_checker_gpu = Arc::new(RangeCheckerGpuTraceGenerator::<8>::default());
            let pow_checker_gpu = Arc::new(PowerCheckerGpuTraceGenerator::<2, 32>::default());
            let proof_shape = proof_shape::cuda::ProofShapeChipGpu::<4, 8>::new(
                self.idx_encoder.width(),
                self.min_cached_idx,
                self.max_cached,
                range_checker_gpu.clone(),
                pow_checker_gpu.clone(),
            );
            let blob = ProofShapeBlob;
            let mut ctxs = Vec::with_capacity(3);
            // PERF[jpw]: we avoid par_iter so that kernel launches occur on the same stream.
            // This can be parallelized to separate streams for more CUDA stream parallelism, but it
            // will require recording events so streams properly sync for cudaMemcpyAsync and kernel
            // launches
            ctxs.extend(
                [
                    ProofShapeModuleChipGpu::ProofShape(proof_shape),
                    ProofShapeModuleChipGpu::PublicValues,
                ]
                .iter()
                .map(|chip| chip.generate_trace(child_vk, proofs, preflights, &blob))
                .collect::<Vec<_>>(),
            );
            // Caution: proof_shape **must** finish trace gen before range_checker or pow_checker
            // can start trace gen with the correct multiplicities
            ctxs.push(Arc::try_unwrap(range_checker_gpu).unwrap().generate_trace());
            let pow_trace = Arc::try_unwrap(pow_checker_gpu).unwrap().generate_trace();
            accumulate_power_checker_counts_from_gpu(&pow_trace, &self.pow_checker);

            ctxs.into_iter()
                .map(AirProvingContextV2::simple_no_pis)
                .collect()
        }
    }

    enum ProofShapeModuleChipGpu {
        ProofShape(proof_shape::cuda::ProofShapeChipGpu<4, 8>),
        PublicValues,
    }

    impl ModuleChip<GlobalCtxGpu, GpuBackendV2> for ProofShapeModuleChipGpu {
        // Empty blob so no difference in type between cpu and gpu
        type ModuleSpecificCtx = ProofShapeBlob;

        fn generate_trace(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            _: &ProofShapeBlob,
        ) -> DeviceMatrix<F> {
            use ProofShapeModuleChipGpu::*;
            match self {
                ProofShape(chip) => chip.generate_trace(child_vk, preflights),
                PublicValues => pvs::cuda::generate_trace(proofs, preflights),
            }
        }
    }

    fn accumulate_power_checker_counts_from_gpu(
        pow_trace: &DeviceMatrix<F>,
        cpu_checker: &Arc<PowerCheckerTraceGenerator<2, 32>>,
    ) {
        // Each column is stored contiguously: [log | pow | mult_pow | mult_range]
        let height = pow_trace.height();
        let host = pow_trace
            .buffer()
            .to_host()
            .expect("failed to copy power checker trace to host");
        for row in 0..height {
            let log = host[row].as_canonical_u32() as usize;
            let mult_pow = host[row + 2 * height].as_canonical_u32();
            if mult_pow != 0 {
                cpu_checker.add_pow_count(log, mult_pow);
            }
            let mult_range = host[row + 3 * height].as_canonical_u32();
            if mult_range != 0 {
                cpu_checker.add_range_count(log, mult_range);
            }
        }
    }
}
