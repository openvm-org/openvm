use std::array::from_fn;

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
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use stark_recursion_circuit_derive::AlignedBorrow;

pub mod bus;
pub mod commit;
pub mod memory;
pub mod verifier;
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct RootVerifierPvs<F> {
    /// Hashed combination of the app-level ProgramAir cached trace, the Merkle root commit of
    /// the starting app memory state (i.e. initial_root), and the initial app program counter
    /// (i.e. initial_pc).
    pub app_exe_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpressionAir, which is
    /// derived from the app_vk
    pub app_dag_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpressionAir,
    /// which is derived from the leaf_vk
    pub leaf_dag_commit: [F; DIGEST_SIZE],
    /// Cached trace commit of the first (i.e. index 0) internal-recursive layer verifier
    /// circuit's SymbolicExpressionAir, which is derived from the internal_for_leaf_vk
    pub internal_for_leaf_dag_commit: [F; DIGEST_SIZE],
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
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone());
        let (memory_ctx, memory_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        (
            vec![commit_ctx, memory_ctx],
            commit_inputs
                .into_iter()
                .chain(memory_inputs.into_iter())
                .collect_vec(),
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
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone());
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
            commit_inputs
                .into_iter()
                .chain(memory_inputs.into_iter())
                .collect_vec(),
        )
    }
}

pub(in crate::circuit::root) fn digests_to_poseidon2_input<T: Clone>(
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

pub(in crate::circuit::root) fn pad_slice_to_poseidon2_input<T: Clone>(
    x: &[T],
    fill: T,
) -> [T; POSEIDON2_WIDTH] {
    from_fn(|i| {
        if i < x.len() {
            x[i].clone()
        } else {
            fill.clone()
        }
    })
}
