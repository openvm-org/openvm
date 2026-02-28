use std::{borrow::Borrow, sync::Arc};

use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
    AirRef,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use recursion_circuit::system::AggregationSubCircuit;
use verify_stark::pvs::{VerifierBasePvs, VERIFIER_PVS_AIR_ID};

use crate::{prover::Circuit, SC};

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    };
}

pub mod bus;
pub mod receiver;
pub mod verifier;
pub mod vm_pvs;

#[derive(derive_new::new, Clone)]
pub struct NonRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> Circuit for NonRootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let pvs_air_consistency_bus_idx = self.verifier_circuit.next_bus_idx();
        let public_values_bus = bus_inventory.public_values_bus;
        [Arc::new(verifier::VerifierPvsAir {
            public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            pvs_air_consistency_bus: bus::PvsAirConsistencyBus::new(pvs_air_consistency_bus_idx),
            deferral_config: verifier::VerifierDeferralConfig::Disabled,
        }) as AirRef<SC>]
        .into_iter()
        .chain([Arc::new(vm_pvs::VmPvsAir {
            public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            pvs_air_consistency_bus: bus::PvsAirConsistencyBus::new(pvs_air_consistency_bus_idx),
            deferral_enabled: false,
        }) as AirRef<SC>])
        .chain([Arc::new(receiver::UserPvsReceiverAir { public_values_bus }) as AirRef<SC>])
        .chain(self.verifier_circuit.airs())
        .collect_vec()
    }
}

// Trait that non-root and compression provers use to remain generic in PB
pub trait NonRootTraceGen<PB: ProverBackend> {
    fn new() -> Self;
    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
        child_dag_commit: PB::Commitment,
    ) -> AirProvingContext<PB>;
    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
    ) -> Vec<AirProvingContext<PB>>;
}

pub struct NonRootTraceGenImpl;

fn derive_child_level(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_app: bool,
) -> verifier::VerifierChildLevel {
    if child_is_app {
        verifier::VerifierChildLevel::App
    } else {
        let child_pvs: &VerifierBasePvs<F> = proofs[0].public_values[VERIFIER_PVS_AIR_ID]
            .as_slice()
            .borrow();
        match child_pvs.internal_flag {
            F::ZERO => verifier::VerifierChildLevel::Leaf,
            F::ONE => verifier::VerifierChildLevel::InternalForLeaf,
            F::TWO => verifier::VerifierChildLevel::InternalRecursive,
            _ => unreachable!(),
        }
    }
}

impl NonRootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for NonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
        let child_level = derive_child_level(proofs, child_is_app);
        verifier::generate_proving_ctx(proofs, child_level, child_dag_commit, false).0
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
    ) -> Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>> {
        vec![
            vm_pvs::generate_proving_ctx(proofs, child_is_app, false),
            receiver::generate_proving_ctx(proofs, child_is_app),
        ]
    }
}

#[cfg(feature = "cuda")]
impl NonRootTraceGen<GpuBackend> for NonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContext<GpuBackend> {
        let child_level = derive_child_level(proofs, child_is_app);
        let cpu_ctx =
            verifier::generate_proving_ctx(proofs, child_level, child_dag_commit, false).0;
        transport_air_proving_ctx_to_device(cpu_ctx)
    }

    fn generate_other_proving_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_app: bool,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let vm_cpu_ctx = vm_pvs::generate_proving_ctx(proofs, child_is_app, false);
        let receiver_cpu_ctx = receiver::generate_proving_ctx(proofs, child_is_app);
        vec![
            transport_air_proving_ctx_to_device(vm_cpu_ctx),
            transport_air_proving_ctx_to_device(receiver_cpu_ctx),
        ]
    }
}
