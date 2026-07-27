use std::borrow::Borrow;

use halo2_base::halo2_proofs::arithmetic::Field;
use openvm_continuations::circuit::root::{RootVerifierPvs, USER_PVS_COMMIT_AIR_ID};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE as APP_DIGEST_SIZE;
use openvm_verify_stark_host::pvs::VERIFIER_PVS_AIR_ID;

use crate::{
    chip_traits::GateInst,
    field::baby_bear::{ReducedBabyBearWire, BABY_BEAR_MODULUS_U64},
    stages::full_pipeline::ProofWire,
    Fr,
};

#[repr(C)]
pub struct StaticVerifierPvs<T> {
    /// Hashed combination of the app-level ProgramAir cached trace, the Merkle root commit of
    /// the starting app memory state (i.e. initial_root), and the initial app program counter
    /// (i.e. initial_pc).
    pub app_exe_commit: T,
    /// Commit to the app-level verifying key, computed by hashing the cached_commit and
    /// vk_pre_hash components of the app, leaf, and internal-for-leaf vk commits.
    pub app_vm_commit: T,
    /// The number of user public values is a configuration parameter in the App VM. This parameter
    /// is treated as a constant in the static verifier circuit.
    pub user_public_values: Vec<T>,
}

impl<T: Clone> StaticVerifierPvs<T> {
    pub fn to_vec(&self) -> Vec<T> {
        let mut vec = vec![self.app_exe_commit.clone(), self.app_vm_commit.clone()];
        vec.extend_from_slice(&self.user_public_values);
        vec
    }

    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            app_exe_commit: slice[0].clone(),
            app_vm_commit: slice[1].clone(),
            user_public_values: slice[2..].to_vec(),
        }
    }
}

/// Extracts the public values from the root proof and returns them. These public values will be
/// re-exposed as public values of the static verifier circuit, but that is **not** done in this
/// function.
pub fn extract_public_values<B: GateInst>(
    b: &mut B,
    proof: &ProofWire<B::F>,
) -> StaticVerifierPvs<B::F> {
    let root_pvs: &RootVerifierPvs<ReducedBabyBearWire<B::F>> =
        proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();
    let app_exe_commit = compress_babybear_wires_to_bn254(b, root_pvs.app_exe_commit);
    let app_vm_commit = compress_babybear_wires_to_bn254(b, root_pvs.app_vm_commit);
    let user_pvs = &proof.public_values[USER_PVS_COMMIT_AIR_ID];
    let user_public_values = user_pvs.iter().map(|bb| bb.value()).collect::<Vec<_>>();

    StaticVerifierPvs {
        app_exe_commit,
        app_vm_commit,
        user_public_values,
    }
}

pub fn compress_babybear_wires_to_bn254<B: GateInst>(
    b: &mut B,
    base_elts: [ReducedBabyBearWire<B::F>; APP_DIGEST_SIZE],
) -> B::F {
    let reduced_elts = base_elts.map(|bb| bb.value());
    let order = Fr::from(BABY_BEAR_MODULUS_U64);
    let mut bases = [Fr::ONE; APP_DIGEST_SIZE];
    for i in 1..APP_DIGEST_SIZE {
        bases[i] = bases[i - 1] * order;
    }
    b.inner_product_const(&reduced_elts, &bases)
}
