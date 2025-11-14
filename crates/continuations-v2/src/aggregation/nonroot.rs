use std::sync::Arc;

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use recursion_circuit::system::VerifierSubCircuit;
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, DIGEST_SIZE, F, StarkEngineV2, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    poseidon2::sponge::{DuplexSponge, DuplexSpongeRecorder},
    proof::Proof,
    prover::{CommittedTraceDataV2, CpuBackendV2, DeviceDataTransporterV2, ProvingContextV2},
};

use crate::{
    aggregation::{
        AggregationCircuit, AggregationProver, MAX_NUM_PROOFS, trace_heights_tracing_info,
    },
    public_values::{
        receiver::{self, UserPvsReceiverAir},
        verifier::{self, VerifierPvsAir},
    },
};

#[derive(Clone, Debug)]
pub struct NonRootStarkProof {
    pub inner: Proof,
    pub user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
}

/*
 * Stateless struct to generate the AIRs of the aggregation circuit
 */
#[derive(derive_new::new, Clone)]
pub struct NonRootAggregationCircuit {
    pub verifier_circuit: Arc<VerifierSubCircuit<MAX_NUM_PROOFS>>,
}

impl AggregationCircuit for NonRootAggregationCircuit {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        airs(&self.verifier_circuit)
    }
}

/*
 * Struct to generate an aggregation proof
 */
pub struct NonRootAggregationProver<const NUM_CHILDREN: usize> {
    pk: Arc<MultiStarkProvingKeyV2>,
    vk: Arc<MultiStarkVerifyingKeyV2>,

    // TODO: tracegen currently requires storing these, we should revisit this
    child_vk: Arc<MultiStarkVerifyingKeyV2>,
    child_vk_pcs_data: CommittedTraceDataV2<CpuBackendV2>,
    circuit: Arc<NonRootAggregationCircuit>,
}

impl<const NUM_CHILDREN: usize> AggregationProver<CpuBackendV2>
    for NonRootAggregationProver<NUM_CHILDREN>
{
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKeyV2> {
        self.vk.clone()
    }

    fn generate_proving_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> ProvingContextV2<CpuBackendV2> {
        assert!(proofs.len() <= NUM_CHILDREN);
        ProvingContextV2 {
            per_trace: vec![verifier::generate_proving_ctx(proofs, user_pv_commit)]
                .into_iter()
                .chain(
                    self.circuit
                        .verifier_circuit
                        .generate_proving_ctxs::<DuplexSpongeRecorder>(
                            &self.child_vk,
                            self.child_vk_pcs_data.clone(),
                            proofs,
                        ),
                )
                .chain([receiver::generate_proving_ctx(
                    proofs,
                    user_pv_commit.is_some(),
                )])
                .enumerate()
                .collect_vec(),
        }
    }

    fn agg_prove(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Result<Proof> {
        let ctx = self.generate_proving_ctx(proofs, user_pv_commit);
        if tracing::enabled!(tracing::Level::INFO) {
            trace_heights_tracing_info(&ctx.per_trace, &airs(&self.circuit.verifier_circuit));
        }
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(self.pk.params);
        Ok(engine.prove(
            &engine.device().transport_pk_to_device(self.pk.as_ref()),
            ctx,
        ))
    }
}

impl<const NUM_CHILDREN: usize> NonRootAggregationProver<NUM_CHILDREN> {
    pub fn new(child_vk: Arc<MultiStarkVerifyingKeyV2>, system_params: SystemParams) -> Self {
        let verifier_circuit =
            VerifierSubCircuit::new_with_set_continuations(child_vk.clone(), true);
        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let (pk, vk) = engine.keygen(&airs(&verifier_circuit));
        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            child_vk,
            child_vk_pcs_data,
            circuit: Arc::new(NonRootAggregationCircuit::new(Arc::new(verifier_circuit))),
        }
    }

    pub fn get_circuit(&self) -> Arc<NonRootAggregationCircuit> {
        self.circuit.clone()
    }
}

fn airs(
    verifier_circuit: &VerifierSubCircuit<MAX_NUM_PROOFS>,
) -> Vec<AirRef<BabyBearPoseidon2Config>> {
    let public_values_bus = verifier_circuit.bus_inventory.public_values_bus;
    [Arc::new(VerifierPvsAir {
        public_values_bus,
        cached_commit_bus: verifier_circuit.bus_inventory.cached_commit_bus,
    }) as AirRef<BabyBearPoseidon2Config>]
    .into_iter()
    .chain(verifier_circuit.airs())
    .chain([Arc::new(UserPvsReceiverAir { public_values_bus }) as AirRef<BabyBearPoseidon2Config>])
    .collect_vec()
}
