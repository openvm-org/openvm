use core::cmp::Reverse;
use std::sync::Arc;

use itertools::izip;
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::Proof,
};

use crate::{
    primitives::{
        bus::{PowerCheckerBus, RangeCheckerBus},
        pow::{PowerCheckerAir, PowerCheckerTraceGenerator},
        range::{RangeCheckerAir, RangeCheckerTraceGenerator},
    },
    proof_shape::{
        air::ProofShapeAir,
        bus::{NumPublicValuesBus, ProofShapePermutationBus},
        pvs::PublicValuesAir,
    },
    system::{AirModule, BusIndexManager, BusInventory, Preflight, ProofShapePreflight},
};

pub mod air;
pub mod bus;
pub mod pvs;

pub struct ProofShapeModule {
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
}

impl<TS: FiatShamirTranscript + TranscriptHistory> AirModule<TS> for ProofShapeModule {
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

    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS) {
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

    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        let range_checker = Arc::new(RangeCheckerTraceGenerator::<8>::default());
        let pow_checker = Arc::new(PowerCheckerTraceGenerator::<2, 32>::default());
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(air::generate_trace::<4, 8>(
                    &self.mvk,
                    proof,
                    preflight,
                    self.idx_encoder.clone(),
                    self.min_cached_idx,
                    self.max_cached,
                    range_checker.clone(),
                    pow_checker.clone(),
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(pvs::generate_trace(proof, preflight))),
                public_values: vec![],
            },
            range_checker.generate_proof_input(),
            pow_checker.generate_proof_input(),
        ]
    }
}
