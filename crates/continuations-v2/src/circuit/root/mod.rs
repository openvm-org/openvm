use std::{array::from_fn, borrow::Borrow, sync::Arc};

use itertools::Itertools;
use openvm_circuit::{
    arch::POSEIDON2_WIDTH,
    system::memory::{dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof},
};
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
    AirRef,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use recursion_circuit::system::AggregationSubCircuit;
use stark_recursion_circuit_derive::AlignedBorrow;
use verify_stark::pvs::{DeferralPvs, DEF_PVS_AIR_ID};

use super::user_pvs::{commit, memory};
use crate::{
    bn254::CommitBytes,
    circuit::{
        deferral::verify::{
            bus::{DeferralAccPathBus, MemoryMerkleRootsBus},
            paths::AccMerklePathsAir,
            DeferralMerkleProofs,
        },
        Circuit,
    },
    SC,
};

pub mod bus;
pub mod verifier;

#[derive(derive_new::new, Clone)]
pub struct RootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    pub(crate) internal_recursive_dag_commit: CommitBytes,
    pub(crate) def_hook_commit: Option<CommitBytes>,
    pub(crate) memory_dimensions: MemoryDimensions,
    pub(crate) num_user_pvs: usize,
}

impl<S: AggregationSubCircuit> Circuit for RootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();

        let user_pvs_commit_bus = bus::UserPvsCommitBus::new(next_bus_idx);
        let user_pvs_commit_tree_bus = bus::UserPvsCommitTreeBus::new(next_bus_idx + 1);
        let memory_merkle_commit_bus = bus::MemoryMerkleCommitBus::new(next_bus_idx + 2);
        let def_acc_paths_bus = DeferralAccPathBus::new(next_bus_idx + 3);
        let memory_merkle_roots_bus = MemoryMerkleRootsBus::new(next_bus_idx + 4);

        let verifier_pvs_air = verifier::RootVerifierPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            memory_merkle_commit_bus,
            def_acc_paths_bus,
            memory_merkle_roots_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
            expected_def_hook_commit: self.def_hook_commit,
        };
        let user_pvs_commit_air = commit::UserPvsCommitAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            None,
            self.num_user_pvs,
        );
        let user_pvs_memory_air = memory::UserPvsInMemoryAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
            self.memory_dimensions,
            self.num_user_pvs,
        );
        let acc_paths_air = self.def_hook_commit.map(|_| {
            Arc::new(AccMerklePathsAir::new(
                bus_inventory.poseidon2_compress_bus,
                def_acc_paths_bus,
                memory_merkle_roots_bus,
                self.memory_dimensions,
            )) as AirRef<SC>
        });

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain([Arc::new(user_pvs_commit_air) as AirRef<SC>])
            .chain([Arc::new(user_pvs_memory_air) as AirRef<SC>])
            .chain(self.verifier_circuit.airs())
            .chain(acc_paths_air)
            .collect_vec()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct RootVerifierPvs<F> {
    /// Hashed combination of the app-level ProgramAir cached trace, the Merkle root commit of
    /// the starting app memory state (i.e. initial_root), and the initial app program counter
    /// (i.e. initial_pc).
    pub app_exe_commit: [F; DIGEST_SIZE],
    /// Commit to the app-level verifying key, computed by compressing together the app, leaf,
    /// and internal-for-leaf DAG commits.
    pub app_vk_commit: [F; DIGEST_SIZE],
}

// Trait that root provers use to remain generic in PB. Tracegen returns both the AIR proving
// contexts and the Poseidon2 compress inputs that are to be fed to Poseidon2Air.
pub trait RootTraceGen<PB: ProverBackend> {
    fn new(deferral_enabled: bool) -> Self;
    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        memory_dimensions: MemoryDimensions,
    ) -> (Vec<AirProvingContext<PB>>, Vec<[PB::Val; POSEIDON2_WIDTH]>);
    fn generate_other_proving_ctxs(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<PB::Val>>,
    ) -> (Vec<AirProvingContext<PB>>, Vec<[PB::Val; POSEIDON2_WIDTH]>);
}

pub struct RootTraceGenImpl {
    pub deferral_enabled: bool,
}

impl RootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for RootTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> (
        Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (verifier_ctx, verifier_p2_inputs) =
            verifier::generate_proving_ctx(proof, self.deferral_enabled);
        let (commit_ctx, commit_inputs) =
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone(), true);
        let (memory_ctx, memory_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        (
            vec![verifier_ctx, commit_ctx, memory_ctx],
            verifier_p2_inputs
                .into_iter()
                .chain(commit_inputs)
                .chain(memory_inputs)
                .collect_vec(),
        )
    }

    fn generate_other_proving_ctxs(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> (
        Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (paths_ctx, paths_inputs) = if let Some(deferral_merkle_proofs) = deferral_merkle_proofs
        {
            assert!(self.deferral_enabled);
            let def_pvs: &DeferralPvs<F> = proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow();
            let depth = def_pvs.depth.as_canonical_u32() as usize;
            let (ctx, inputs) = crate::circuit::deferral::verify::paths::generate_proving_input(
                def_pvs.initial_acc_hash,
                def_pvs.final_acc_hash,
                &deferral_merkle_proofs.initial_merkle_proof,
                &deferral_merkle_proofs.final_merkle_proof,
                memory_dimensions,
                depth,
                depth == 0,
            );
            (Some(ctx), inputs)
        } else {
            assert!(!self.deferral_enabled);
            (None, vec![])
        };
        (paths_ctx.into_iter().collect_vec(), paths_inputs)
    }
}

#[cfg(feature = "cuda")]
impl RootTraceGen<GpuBackend> for RootTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> (
        Vec<AirProvingContext<GpuBackend>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (cpu_ctxs, inputs) =
            self.generate_pre_verifier_subcircuit_ctx(proof, user_pvs_proof, memory_dimensions);
        let gpu_ctxs = cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device)
            .collect_vec();
        (gpu_ctxs, inputs)
    }

    fn generate_other_proving_ctxs(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> (
        Vec<AirProvingContext<GpuBackend>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (cpu_ctxs, inputs) =
            self.generate_other_proving_ctxs(proof, memory_dimensions, deferral_merkle_proofs);
        let gpu_ctxs = cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device)
            .collect_vec();
        (gpu_ctxs, inputs)
    }
}

pub(crate) fn digests_to_poseidon2_input<T: Clone>(
    x: [T; DIGEST_SIZE],
    y: [T; DIGEST_SIZE],
) -> [T; POSEIDON2_WIDTH] {
    from_fn(|i| {
        if i < DIGEST_SIZE {
            x[i].clone()
        } else {
            y[i - DIGEST_SIZE].clone()
        }
    })
}

pub fn poseidon2_input_to_digests<T>(
    x: [T; POSEIDON2_WIDTH],
) -> ([T; DIGEST_SIZE], [T; DIGEST_SIZE]) {
    let mut it = x.into_iter();
    let commit = from_fn(|_| it.next().unwrap());
    let len = from_fn(|_| it.next().unwrap());
    (commit, len)
}

pub(crate) fn pad_slice_to_poseidon2_input<T: Clone>(x: &[T], fill: T) -> [T; POSEIDON2_WIDTH] {
    from_fn(|i| {
        if i < x.len() {
            x[i].clone()
        } else {
            fill.clone()
        }
    })
}

pub(crate) fn zero_hash(depth: usize) -> [F; DIGEST_SIZE] {
    let mut ret = [F::ZERO; DIGEST_SIZE];
    for _ in 0..depth {
        ret = poseidon2_compress_with_capacity(ret, ret).0;
    }
    ret
}
