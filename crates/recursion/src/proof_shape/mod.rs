use core::cmp::Reverse;
use std::sync::Arc;

use itertools::izip;
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::Proof,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
};

use crate::{
    primitives::{
        bus::{PowerCheckerBus, RangeCheckerBus},
        pow::{PowerCheckerAir, PowerCheckerTraceGenerator},
        range::{RangeCheckerAir, RangeCheckerTraceGenerator},
    },
    proof_shape::{
        bus::{NumPublicValuesBus, ProofShapePermutationBus},
        proof_shape::ProofShapeAir,
        pvs::PublicValuesAir,
    },
    system::{
        AirModule, BusIndexManager, BusInventory, GlobalCtxCpu, ModuleChip, Preflight,
        ProofShapePreflight, TraceGenModule,
    },
};

pub mod bus;
pub mod proof_shape;
pub mod pvs;

#[cfg(feature = "cuda")]
mod cuda_abi;

pub struct ProofShapeModule {
    // TODO[jpw]: remove and only store relevant parts to allow recursion
    mvk: Arc<MultiStarkVerifyingKeyV2>,
    bus_inventory: BusInventory,

    range_bus: RangeCheckerBus,
    pow_bus: PowerCheckerBus,

    permutation_bus: ProofShapePermutationBus,
    num_pvs_bus: NumPublicValuesBus,

    // Required for ProofShapeAir tracegen + constraints
    idx_encoder: Arc<Encoder>,
    min_cached_idx: usize,
    max_cached: usize,
}

impl ProofShapeModule {
    pub fn new(
        mvk: Arc<MultiStarkVerifyingKeyV2>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        let idx_encoder = Arc::new(Encoder::new(mvk.inner.per_air.len(), 2, true));

        let (min_cached_idx, min_cached) = mvk
            .inner
            .per_air
            .iter()
            .enumerate()
            .min_by_key(|(_, avk)| avk.params.width.cached_mains.len())
            .map(|(idx, avk)| (idx, avk.params.width.cached_mains.len()))
            .unwrap();
        let mut max_cached = mvk
            .inner
            .per_air
            .iter()
            .map(|avk| avk.params.width.cached_mains.len())
            .max()
            .unwrap();
        if min_cached == max_cached {
            max_cached += 1;
        }

        Self {
            mvk,
            bus_inventory,
            range_bus: RangeCheckerBus::new(b.new_bus_idx()),
            pow_bus: PowerCheckerBus::new(b.new_bus_idx()),
            permutation_bus: ProofShapePermutationBus::new(b.new_bus_idx()),
            num_pvs_bus: NumPublicValuesBus::new(b.new_bus_idx()),
            idx_encoder,
            min_cached_idx,
            max_cached,
        }
    }

    pub fn run_preflight<TS>(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS)
    where
        TS: FiatShamirTranscript + TranscriptHistory,
    {
        ts.observe_commit(self.mvk.pre_hash);
        ts.observe_commit(proof.common_main_commit);

        let vk = &self.mvk.inner;

        let mut pvs_tidx = vec![];

        for (trace_vdata, avk, pvs) in izip!(&proof.trace_vdata, &vk.per_air, &proof.public_values)
        {
            let is_air_present = trace_vdata.is_some();

            if !avk.is_required {
                ts.observe(F::from_bool(is_air_present));
            }
            if let Some(trace_vdata) = trace_vdata {
                if let Some(pdata) = avk.preprocessed_data.as_ref() {
                    ts.observe_commit(pdata.commit);
                } else {
                    ts.observe(F::from_canonical_usize(trace_vdata.hypercube_dim));
                }
                debug_assert_eq!(
                    avk.params.width.cached_mains.len(),
                    trace_vdata.cached_commitments.len()
                );
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
        sorted_trace_vdata.sort_by_key(|(_, data)| Reverse(data.hypercube_dim));

        let n_max = proof
            .trace_vdata
            .iter()
            .flat_map(|datum| datum.as_ref().map(|datum| datum.hypercube_dim))
            .max()
            .unwrap();
        let num_layers = proof.gkr_proof.claims_per_layer.len();
        let l_skip = vk.params.l_skip;
        let n_logup = num_layers.saturating_sub(l_skip);

        preflight.proof_shape = ProofShapePreflight {
            sorted_trace_vdata,
            pvs_tidx,
            post_tidx: ts.len(),
            n_max,
            n_logup,
            l_skip,
        };
    }
}

impl AirModule for ProofShapeModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let proof_shape_air = ProofShapeAir::<4, 8> {
            vk: self.mvk.clone(),
            min_cached_idx: self.min_cached_idx,
            max_cached: self.max_cached,
            idx_encoder: self.idx_encoder.clone(),
            range_bus: self.range_bus,
            pow_bus: self.pow_bus,
            permutation_bus: self.permutation_bus,
            num_pvs_bus: self.num_pvs_bus,
            gkr_module_bus: self.bus_inventory.gkr_module_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
            air_heights_bus: self.bus_inventory.air_heights_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        let pvs_air = PublicValuesAir {
            _public_values_bus: self.bus_inventory.public_values_bus,
            num_pvs_bus: self.num_pvs_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        let range_checker = RangeCheckerAir::<8> {
            bus: self.range_bus,
        };
        let pow_checker = PowerCheckerAir::<2, 32> {
            pow_bus: self.pow_bus,
            range_bus: self.range_bus,
        };
        vec![
            Arc::new(proof_shape_air) as AirRef<_>,
            Arc::new(pvs_air) as AirRef<_>,
            Arc::new(range_checker) as AirRef<_>,
            Arc::new(pow_checker) as AirRef<_>,
        ]
    }
}

/// Empty blob
struct ProofShapeBlob;

impl TraceGenModule<GlobalCtxCpu, CpuBackendV2> for ProofShapeModule {
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        let range_checker = Arc::new(RangeCheckerTraceGenerator::<8>::default());
        let pow_checker = Arc::new(PowerCheckerTraceGenerator::<2, 32>::default());
        let proof_shape = proof_shape::ProofShapeChip::<4, 8>::new(
            self.idx_encoder.clone(),
            self.min_cached_idx,
            self.max_cached,
            range_checker.clone(),
            pow_checker.clone(),
        );
        let blob = ProofShapeBlob;
        let mut ctxs = Vec::with_capacity(4);
        // Caution: proof_shape **must** finish trace gen before range_checker or pow_checker can
        // start trace gen with the correct multiplicities
        ctxs.extend(
            [
                ProofShapeModuleChip::ProofShape(proof_shape),
                ProofShapeModuleChip::PublicValues,
            ]
            .par_iter()
            .map(|chip| chip.generate_trace(child_vk, proofs, preflights, &blob))
            .collect::<Vec<_>>(),
        );
        ctxs.extend(
            [
                range_checker.generate_trace_row_major(),
                pow_checker.generate_trace_row_major(),
            ]
            .map(|trace| ColMajorMatrix::from_row_major(&trace)),
        );
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

    use super::*;
    use crate::{
        cuda::{GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
        primitives::{
            pow::cuda::PowerCheckerGpuTraceGenerator, range::cuda::RangeCheckerGpuTraceGenerator,
        },
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackendV2> for ProofShapeModule {
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
        ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
            let range_checker = Arc::new(RangeCheckerGpuTraceGenerator::<8>::default());
            let pow_checker = Arc::new(PowerCheckerGpuTraceGenerator::<2, 32>::default());
            let proof_shape = proof_shape::cuda::ProofShapeChipGpu::<4, 8>::new(
                self.idx_encoder.width(),
                self.min_cached_idx,
                self.max_cached,
                range_checker.clone(),
                pow_checker.clone(),
            );
            let blob = ProofShapeBlob;
            let mut ctxs = Vec::with_capacity(4);
            // Caution: proof_shape **must** finish trace gen before range_checker or pow_checker
            // can start trace gen with the correct multiplicities
            ctxs.extend(
                [
                    ProofShapeModuleChipGpu::ProofShape(proof_shape),
                    ProofShapeModuleChipGpu::PublicValues,
                ]
                .par_iter()
                .map(|chip| chip.generate_trace(child_vk, proofs, preflights, &blob))
                .collect::<Vec<_>>(),
            );
            ctxs.extend([
                Arc::try_unwrap(range_checker).unwrap().generate_trace(),
                Arc::try_unwrap(pow_checker).unwrap().generate_trace(),
            ]);

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
}
