use std::{array::from_fn, sync::Arc};

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
use p3_field::PrimeCharacteristicRing;
use recursion_circuit::system::AggregationSubCircuit;
use stark_recursion_circuit_derive::AlignedBorrow;

use super::user_pvs::{commit, memory};
use crate::{bn254::CommitBytes, circuit::Circuit, SC};

pub mod bus;
pub mod verifier;

#[derive(derive_new::new, Clone)]
pub struct RootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    pub(crate) internal_recursive_dag_commit: CommitBytes,
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

        let verifier_pvs_air = verifier::RootVerifierPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            memory_merkle_commit_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
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

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain([Arc::new(user_pvs_commit_air) as AirRef<SC>])
            .chain([Arc::new(user_pvs_memory_air) as AirRef<SC>])
            .chain(self.verifier_circuit.airs())
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
    fn new() -> Self;
    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> (AirProvingContext<PB>, Vec<[PB::Val; POSEIDON2_WIDTH]>);
    fn generate_other_proving_ctxs(
        &self,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        memory_dimensions: MemoryDimensions,
    ) -> (Vec<AirProvingContext<PB>>, Vec<[PB::Val; POSEIDON2_WIDTH]>);
}

pub struct RootTraceGenImpl;

impl RootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for RootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> (
        AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        verifier::generate_proving_ctx(proof)
    }

    fn generate_other_proving_ctxs(
        &self,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> (
        Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (commit_ctx, commit_inputs) =
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone(), true);
        let (memory_ctx, memory_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        (
            vec![commit_ctx, memory_ctx],
            commit_inputs.into_iter().chain(memory_inputs).collect_vec(),
        )
    }
}

#[cfg(feature = "cuda")]
impl RootTraceGen<GpuBackend> for RootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> (AirProvingContext<GpuBackend>, Vec<[F; POSEIDON2_WIDTH]>) {
        let (cpu_ctx, inputs) = verifier::generate_proving_ctx(proof);
        (transport_air_proving_ctx_to_device(cpu_ctx), inputs)
    }

    fn generate_other_proving_ctxs(
        &self,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> (
        Vec<AirProvingContext<GpuBackend>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (commit_cpu_ctx, commit_inputs) =
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone(), true);
        let (memory_cpu_ctx, memory_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        (
            vec![
                transport_air_proving_ctx_to_device(commit_cpu_ctx),
                transport_air_proving_ctx_to_device(memory_cpu_ctx),
            ],
            commit_inputs.into_iter().chain(memory_inputs).collect_vec(),
        )
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
