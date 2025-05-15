use std::{array, borrow::BorrowMut};

use derivative::Derivative;
use openvm_circuit::circuit_derive::AlignedBorrow;
use openvm_native_compiler::{
    ir::{Builder, Config, Felt},
    prelude::DIGEST_SIZE,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    openvm_stark_backend::{
        config::{Com, StarkGenericConfig, Val},
        p3_field::PrimeField32,
        proof::Proof,
    },
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use static_assertions::assert_impl_all;

use crate::{verifier::common::types::VmVerifierPvs, SC};

/// Input for the leaf VM verifier.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct InternalVmVerifierInput<SC: StarkGenericConfig> {
    pub self_program_commit: [Val<SC>; DIGEST_SIZE],
    /// The proofs of leaf verifier or internal verifier in the execution order.
    pub proofs: Vec<Proof<SC>>,
}
assert_impl_all!(InternalVmVerifierInput<BabyBearPoseidon2Config>: Serialize, DeserializeOwned);

/// A proof which can prove OpenVM program execution.
#[derive(Deserialize, Serialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct VmStarkProof<SC: StarkGenericConfig> {
    pub proof: Proof<SC>,
    pub user_public_values: Vec<Val<SC>>,
}
assert_impl_all!(VmStarkProof<BabyBearPoseidon2Config>: Serialize, DeserializeOwned);

/// Aggregated state of all segments
#[derive(Debug, Clone, Copy, AlignedBorrow)]
#[repr(C)]
pub struct InternalVmVerifierPvs<T> {
    pub vm_verifier_pvs: VmVerifierPvs<T>,
    pub extra_pvs: InternalVmVerifierExtraPvs<T>,
}

/// Extra PVs for internal VM verifier except VmVerifierPvs.
#[derive(Debug, Clone, Copy, AlignedBorrow)]
#[repr(C)]
pub struct InternalVmVerifierExtraPvs<T> {
    /// The commitment of the leaf verifier program.
    pub leaf_verifier_commit: [T; DIGEST_SIZE],
    /// For recursion verification, a program need its own commitment, but its own commitment
    /// cannot be hardcoded inside the program itself. So the commitment has to be read from
    /// external and be committed.
    pub internal_program_commit: [T; DIGEST_SIZE],
}

impl InternalVmVerifierInput<SC> {
    pub fn chunk_leaf_or_internal_proofs(
        self_program_commit: [Val<SC>; DIGEST_SIZE],
        proofs: &[Proof<SC>],
        chunk: usize,
    ) -> Vec<Self> {
        proofs
            .chunks(chunk)
            .map(|chunk| Self {
                self_program_commit,
                proofs: chunk.to_vec(),
            })
            .collect()
    }
}

impl<F: PrimeField32> InternalVmVerifierPvs<Felt<F>> {
    pub fn uninit<C: Config<F = F>>(builder: &mut Builder<C>) -> Self {
        Self {
            vm_verifier_pvs: VmVerifierPvs::<Felt<F>>::uninit(builder),
            extra_pvs: InternalVmVerifierExtraPvs::<Felt<F>>::uninit(builder),
        }
    }
}

impl<F: Default + Clone> InternalVmVerifierPvs<Felt<F>> {
    pub fn flatten(self) -> Vec<Felt<F>> {
        let mut v = vec![Felt(0, Default::default()); InternalVmVerifierPvs::<u8>::width()];
        *v.as_mut_slice().borrow_mut() = self;
        v
    }
}

impl<F: PrimeField32> InternalVmVerifierExtraPvs<Felt<F>> {
    pub fn uninit<C: Config<F = F>>(builder: &mut Builder<C>) -> Self {
        Self {
            leaf_verifier_commit: array::from_fn(|_| builder.uninit()),
            internal_program_commit: array::from_fn(|_| builder.uninit()),
        }
    }
}

impl<F: Default + Clone> InternalVmVerifierExtraPvs<Felt<F>> {
    pub fn flatten(self) -> Vec<Felt<F>> {
        let mut v = vec![Felt(0, Default::default()); InternalVmVerifierExtraPvs::<u8>::width()];
        *v.as_mut_slice().borrow_mut() = self;
        v
    }
}
