use std::sync::Arc;

use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
    AirRef,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use recursion_circuit::system::AggregationSubCircuit;
use verify_stark::pvs::{DeferralPvs, VmPvs, DEF_PVS_AIR_ID, VM_PVS_AIR_ID};

use crate::{
    bn254::CommitBytes,
    circuit::{
        nonroot::{
            bus::PvsAirConsistencyBus,
            def_pvs::DeferralPvsAir,
            unset::UnsetPvsAir,
            verifier::{VerifierDeferralConfig, VerifierPvsAir},
        },
        Circuit,
    },
    SC,
};

pub mod app {
    pub use openvm_circuit::arch::{
        CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
        PUBLIC_VALUES_AIR_ID,
    };
}

pub mod bus;
pub mod def_pvs;
pub mod receiver;
pub mod unset;
pub mod verifier;
pub mod vm_pvs;

#[derive(derive_new::new, Clone)]
pub struct NonRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    pub def_hook_commit: Option<CommitBytes>,
}

impl<S: AggregationSubCircuit> Circuit for NonRootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let public_values_bus = bus_inventory.public_values_bus;
        let cached_commit_bus = bus_inventory.cached_commit_bus;
        let poseidon2_bus = bus_inventory.poseidon2_compress_bus;
        let pvs_air_consistency_bus =
            PvsAirConsistencyBus::new(self.verifier_circuit.next_bus_idx());

        let deferral_enabled = self.def_hook_commit.is_some();

        let deferral_config = if deferral_enabled {
            VerifierDeferralConfig::Enabled { poseidon2_bus }
        } else {
            VerifierDeferralConfig::Disabled
        };

        let verifier_pvs_air = Arc::new(VerifierPvsAir {
            public_values_bus,
            cached_commit_bus,
            pvs_air_consistency_bus,
            deferral_config,
        });

        let vm_pvs_air = Arc::new(vm_pvs::VmPvsAir {
            public_values_bus,
            cached_commit_bus,
            pvs_air_consistency_bus,
            deferral_enabled,
        });

        let (idx2_air, other_airs) = if deferral_enabled {
            let def_pvs_air = Arc::new(DeferralPvsAir {
                public_values_bus,
                cached_commit_bus,
                poseidon2_bus,
                pvs_air_consistency_bus,
                expected_def_hook_commit: self.def_hook_commit.clone().unwrap(),
            }) as AirRef<SC>;
            let unset_vm_pvs_air = Arc::new(UnsetPvsAir {
                public_values_bus,
                pvs_air_consistency_bus,
                air_idx: VM_PVS_AIR_ID,
                num_pvs: VmPvs::<u8>::width(),
                def_flag: 1,
            }) as AirRef<SC>;
            let unset_def_pvs_air = Arc::new(UnsetPvsAir {
                public_values_bus,
                pvs_air_consistency_bus,
                air_idx: DEF_PVS_AIR_ID,
                num_pvs: DeferralPvs::<u8>::width(),
                def_flag: 0,
            }) as AirRef<SC>;
            (def_pvs_air, vec![unset_vm_pvs_air, unset_def_pvs_air])
        } else {
            let unset_dummy_air = Arc::new(UnsetPvsAir {
                public_values_bus,
                pvs_air_consistency_bus,
                air_idx: 0,
                num_pvs: 0,
                def_flag: 0,
            }) as AirRef<SC>;
            (unset_dummy_air, vec![])
        };

        [
            verifier_pvs_air as AirRef<SC>,
            vm_pvs_air as AirRef<SC>,
            idx2_air,
        ]
        .into_iter()
        .chain(self.verifier_circuit.airs())
        .chain(other_airs)
        .collect_vec()
    }
}

#[derive(Copy, Clone)]
pub enum ProofsType {
    Vm,
    Deferral,
    Mix,
    Combined,
}

// Trait that non-root and compression provers use to remain generic in PB
pub trait NonRootTraceGen<PB: ProverBackend> {
    fn new(deferral_enabled: bool) -> Self;
    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_dag_commit: PB::Commitment,
    ) -> (Vec<AirProvingContext<PB>>, Vec<[F; POSEIDON2_WIDTH]>);
    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
    ) -> Vec<AirProvingContext<PB>>;
}

pub struct NonRootTraceGenImpl {
    pub deferral_enabled: bool,
}

impl NonRootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for NonRootTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> (
        Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (verifier_pvs_ctx, mut poseidon2_inputs) = verifier::generate_proving_ctx(
            proofs,
            proofs_type,
            child_is_app,
            child_dag_commit,
            self.deferral_enabled,
        );
        let vm_pvs_ctx =
            vm_pvs::generate_proving_ctx(proofs, proofs_type, child_is_app, self.deferral_enabled);

        let idx2_ctx = if self.deferral_enabled {
            let (def_pvs_ctx, def_poseidon2_inputs) =
                def_pvs::generate_proving_ctx(proofs, proofs_type, child_is_app, absent_trace_pvs);
            poseidon2_inputs.extend_from_slice(&def_poseidon2_inputs);
            def_pvs_ctx
        } else {
            unset::generate_proving_ctx(&[], child_is_app)
        };

        (
            vec![verifier_pvs_ctx, vm_pvs_ctx, idx2_ctx],
            poseidon2_inputs,
        )
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
    ) -> Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>> {
        if !self.deferral_enabled {
            return vec![];
        }

        let (vm_unset, def_unset) = match proofs_type {
            ProofsType::Vm => (
                vec![],
                proofs.iter().enumerate().map(|(i, _)| i).collect_vec(),
            ),
            ProofsType::Deferral => (
                proofs.iter().enumerate().map(|(i, _)| i).collect_vec(),
                vec![],
            ),
            ProofsType::Mix => (vec![1], vec![0]),
            ProofsType::Combined => (vec![], vec![]),
        };
        vec![
            unset::generate_proving_ctx(&vm_unset, child_is_app),
            unset::generate_proving_ctx(&def_unset, child_is_app),
        ]
    }
}

#[cfg(feature = "cuda")]
impl NonRootTraceGen<GpuBackend> for NonRootTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_dag_commit: [F; DIGEST_SIZE],
    ) -> (
        Vec<AirProvingContext<GpuBackend>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (cpu_ctxs, poseidon2_inputs) = self.generate_pre_verifier_subcircuit_ctxs(
            proofs,
            proofs_type,
            absent_trace_pvs,
            child_is_app,
            child_dag_commit,
        );
        let gpu_ctxs = cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device)
            .collect_vec();
        (gpu_ctxs, poseidon2_inputs)
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let cpu_ctxs =
            self.generate_post_verifier_subcircuit_ctxs(proofs, proofs_type, child_is_app);
        cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device)
            .collect_vec()
    }
}
