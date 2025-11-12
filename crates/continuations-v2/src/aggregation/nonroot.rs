use std::sync::Arc;

use eyre::Result;
use itertools::Itertools;
use openvm_stark_backend::{
    AirRef,
    prover::{MatrixDimensions, types::AirProofRawInput},
};
use openvm_stark_sdk::{
    config::{
        FriParameters,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::{StarkEngine, StarkFriEngine},
};
use recursion_circuit::system::VerifierSubCircuit;
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, DIGEST_SIZE, F, StarkEngineV2, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    poseidon2::sponge::{DuplexSponge, DuplexSpongeRecorder},
    proof::Proof,
    prover::{
        AirProvingContextV2, CommittedTraceDataV2, CpuBackendV2, DeviceDataTransporterV2,
        ProvingContextV2, StridedColMajorMatrixView,
    },
};

use crate::{
    aggregation::MAX_NUM_PROOFS,
    public_values::{
        receiver::{self, UserPvsReceiverAir},
        verifier::{self, VerifierPvsAir},
    },
};

pub struct NonRootVerifier<const NUM_CHILDREN: usize> {
    pub pk: Arc<MultiStarkProvingKeyV2>,
    pub vk: Arc<MultiStarkVerifyingKeyV2>,

    pub child_vk: Arc<MultiStarkVerifyingKeyV2>,
    child_vk_pcs_data: CommittedTraceDataV2<CpuBackendV2>,

    verifier_circuit: VerifierSubCircuit<MAX_NUM_PROOFS>,
    engine: BabyBearPoseidon2CpuEngineV2<DuplexSponge>,
}

impl<const NUM_CHILDREN: usize> NonRootVerifier<NUM_CHILDREN> {
    pub fn new(child_vk: Arc<MultiStarkVerifyingKeyV2>, system_params: SystemParams) -> Self {
        let verifier_circuit = VerifierSubCircuit::new(child_vk.clone());
        let airs = airs(&verifier_circuit);

        let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let (pk, vk) = engine.keygen(&airs);

        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            child_vk,
            child_vk_pcs_data,
            verifier_circuit,
            engine,
        }
    }

    pub fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        airs(&self.verifier_circuit)
    }

    pub fn generate_proving_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Vec<(usize, AirProvingContextV2<CpuBackendV2>)> {
        assert!(proofs.len() <= NUM_CHILDREN);
        vec![
            verifier::generate_proving_ctx(proofs, user_pv_commit),
            receiver::generate_proving_ctx(proofs, user_pv_commit.is_some()),
        ]
        .into_iter()
        .chain(
            self.verifier_circuit
                .generate_proving_ctxs::<DuplexSpongeRecorder>(
                    &self.child_vk,
                    self.child_vk_pcs_data.clone(),
                    proofs,
                ),
        )
        .enumerate()
        .collect_vec()
    }

    pub fn verify(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Result<Proof> {
        let per_trace = self.generate_proving_ctx(proofs, user_pv_commit);
        trace_heights_tracing_info(&per_trace, &self.airs());
        let proof = self.engine.prove(
            &self
                .engine
                .device()
                .transport_pk_to_device(self.pk.as_ref()),
            ProvingContextV2::new(per_trace),
        );
        Ok(proof)
    }
}

pub fn debug<const NUM_CHILDREN: usize>(
    verifier: NonRootVerifier<NUM_CHILDREN>,
    proofs: &[Proof],
    user_pv_commit: Option<[F; DIGEST_SIZE]>,
) {
    let ctxs = verifier.generate_proving_ctx(proofs, user_pv_commit);
    let transpose = |mat: StridedColMajorMatrixView<F>| Arc::new(mat.to_row_major_matrix());
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let inputs = ctxs
        .iter()
        .map(|(_, ctx)| AirProofRawInput {
            cached_mains: ctx
                .cached_mains
                .iter()
                .map(|cd| transpose(cd.data.mat_view(0)))
                .collect_vec(),
            common_main: Some(transpose(ctx.common_main.as_view().into())),
            public_values: ctx.public_values.clone(),
        })
        .collect_vec();
    let mut keygen_builder = engine.keygen_builder();
    let airs = verifier.airs();
    for air in &airs {
        keygen_builder.add_air(air.clone());
    }
    trace_heights_tracing_info(&ctxs, &airs);
    engine.debug(&airs, &keygen_builder.generate_pk().per_air, &inputs);
}

fn airs(
    verifier_circuit: &VerifierSubCircuit<MAX_NUM_PROOFS>,
) -> Vec<AirRef<BabyBearPoseidon2Config>> {
    let public_values_bus = verifier_circuit.bus_inventory.public_values_bus;
    vec![
        Arc::new(VerifierPvsAir { public_values_bus }) as AirRef<BabyBearPoseidon2Config>,
        Arc::new(UserPvsReceiverAir { public_values_bus }) as AirRef<BabyBearPoseidon2Config>,
    ]
    .into_iter()
    .chain(verifier_circuit.airs())
    .collect_vec()
}

fn trace_heights_tracing_info(
    ctxs: &[(usize, AirProvingContextV2<CpuBackendV2>)],
    airs: &[AirRef<BabyBearPoseidon2Config>],
) {
    if tracing::enabled!(tracing::Level::INFO) {
        for ((_, ctx), air) in ctxs.iter().zip(airs) {
            tracing::info!(
                "{:<40} | Height: {:>8}",
                air.name(),
                ctx.common_main.height()
            );
        }
    }
}
