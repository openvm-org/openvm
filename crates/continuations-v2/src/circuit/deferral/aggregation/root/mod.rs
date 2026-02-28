use std::{borrow::Borrow, sync::Arc};

pub mod bus;
pub mod decommit;
pub mod onion;
pub mod verifier;

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

use crate::{
    bn254::CommitBytes,
    circuit::subair::{MerkleRootBus, MerkleTreeInternalBus, MerkleTreeSubAir},
    prover::Circuit,
    SC,
};

pub type DeferralIoCommit<F> = ([F; DIGEST_SIZE], [F; DIGEST_SIZE]);

#[derive(derive_new::new, Clone)]
pub struct DeferralRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    pub(crate) internal_recursive_dag_commit: CommitBytes,
}

impl<S: AggregationSubCircuit> Circuit for DeferralRootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();
        let io_commit_bus = bus::IoCommitBus::new(next_bus_idx);
        let onion_res_bus = bus::OnionResultBus::new(next_bus_idx + 1);
        let def_vk_commit_bus = bus::DefVkCommitBus::new(next_bus_idx + 2);
        let merkle_root_bus = MerkleRootBus::new(next_bus_idx + 3);
        let merkle_tree_internal_bus = MerkleTreeInternalBus::new(next_bus_idx + 4);

        let verifier_pvs_air = verifier::DeferralRootPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            def_vk_commit_bus,
            merkle_root_bus,
            onion_res_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
        };

        let decommit_air = decommit::MerkleDecommitAir {
            subair: MerkleTreeSubAir::new(
                bus_inventory.poseidon2_compress_bus,
                merkle_root_bus,
                merkle_tree_internal_bus,
                0,
            ),
            io_commit_bus,
        };

        let onion_air = onion::OnionHashAir {
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            def_vk_commit_bus,
            io_commit_bus,
            onion_res_bus,
        };

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain([Arc::new(decommit_air) as AirRef<SC>])
            .chain([Arc::new(onion_air) as AirRef<SC>])
            .chain(self.verifier_circuit.airs())
            .collect_vec()
    }
}

pub struct DeferralRootPreCtx<PB: ProverBackend> {
    pub verifier_pvs_ctx: AirProvingContext<PB>,
    pub decommit_ctx: AirProvingContext<PB>,
    pub onion_ctx: AirProvingContext<PB>,
    pub poseidon2_inputs: Vec<[PB::Val; openvm_circuit::arch::POSEIDON2_WIDTH]>,
}

// Trait used to remain generic in PB.
pub trait DeferralRootTraceGen<PB: ProverBackend> {
    fn new() -> Self;

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<PB::Val>>,
    ) -> DeferralRootPreCtx<PB>;
}

pub struct DeferralRootTraceGenImpl;

fn normalize_leaf_children(
    mut leaf_children: Vec<DeferralIoCommit<F>>,
) -> (Vec<DeferralIoCommit<F>>, usize) {
    assert!(
        !leaf_children.is_empty(),
        "deferral root requires at least one leaf commit"
    );
    let num_real_leaves = leaf_children.len();
    let target_len = leaf_children.len().next_power_of_two();
    leaf_children.resize(target_len, ([F::ZERO; DIGEST_SIZE], [F::ZERO; DIGEST_SIZE]));
    (leaf_children, num_real_leaves)
}

impl DeferralRootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for DeferralRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> DeferralRootPreCtx<CpuBackend<BabyBearPoseidon2Config>> {
        let (leaf_children, num_real_leaves) = normalize_leaf_children(leaf_children);
        let decommit::MerkleDecommitTraceCtx {
            proving_ctx: decommit_ctx,
            poseidon2_inputs: decommit_p2_inputs,
            io_commits,
            merkle_root: computed_merkle_root,
        } = decommit::generate_proving_ctx(leaf_children, num_real_leaves);

        let def_pvs: &crate::circuit::deferral::DeferralAggregationPvs<F> =
            proof.public_values[1].as_slice().borrow();
        assert_eq!(
            computed_merkle_root, def_pvs.merkle_commit,
            "leaf_children do not match the child proof merkle_commit"
        );

        let verifier_pvs: &crate::circuit::deferral::DeferralVerifierPvs<F> =
            proof.public_values[0].as_slice().borrow();
        let (_, def_vk_commit) = verifier::def_vk_commit_from_verifier_pvs(verifier_pvs);

        let onion::OnionTraceCtx {
            proving_ctx: onion_ctx,
            poseidon2_inputs: onion_p2_inputs,
            input_onion,
            output_onion,
        } = onion::generate_proving_ctx(def_vk_commit, io_commits);

        let (verifier_pvs_ctx, verifier_p2_inputs, _) =
            verifier::generate_proving_ctx(proof, input_onion, output_onion);

        DeferralRootPreCtx {
            verifier_pvs_ctx,
            decommit_ctx,
            onion_ctx,
            poseidon2_inputs: verifier_p2_inputs
                .into_iter()
                .chain(decommit_p2_inputs)
                .chain(onion_p2_inputs)
                .collect(),
        }
    }
}

#[cfg(feature = "cuda")]
impl DeferralRootTraceGen<GpuBackend> for DeferralRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> DeferralRootPreCtx<GpuBackend> {
        let DeferralRootPreCtx {
            verifier_pvs_ctx,
            decommit_ctx,
            onion_ctx,
            poseidon2_inputs,
        } = <Self as DeferralRootTraceGen<CpuBackend<BabyBearPoseidon2Config>>>::pre_verifier_subcircuit_tracegen(self, proof, leaf_children);

        DeferralRootPreCtx {
            verifier_pvs_ctx: transport_air_proving_ctx_to_device(verifier_pvs_ctx),
            decommit_ctx: transport_air_proving_ctx_to_device(decommit_ctx),
            onion_ctx: transport_air_proving_ctx_to_device(onion_ctx),
            poseidon2_inputs,
        }
    }
}
