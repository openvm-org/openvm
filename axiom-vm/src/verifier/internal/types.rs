use std::{array, borrow::BorrowMut};

use ax_stark_sdk::ax_stark_backend::{
    config::{Com, StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::types::Proof,
};
use axvm_circuit::circuit_derive::AlignedBorrow;
use axvm_native_compiler::{
    ir::{Builder, Config, Felt},
    prelude::DIGEST_SIZE,
};
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::verifier::common::types::VmVerifierPvs;

#[derive(Debug, AlignedBorrow)]
#[repr(C)]
pub struct InternalVmVerifierPvs<T> {
    pub vm_verifier_pvs: VmVerifierPvs<T>,
    /// The commitment of the leaf verifier program.
    pub leaf_verifier_commit: [T; DIGEST_SIZE],
    /// For recursion verification, a program need its own commitment, but its own commitment cannot
    /// be hardcoded inside the program itself. So the commitment has to be read from external and
    /// be committed.
    pub self_program_commit: [T; DIGEST_SIZE],
}

/// Input for the leaf VM verifier.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct InternalVmVerifierInput<SC: StarkGenericConfig> {
    pub self_program_commit: [Val<SC>; DIGEST_SIZE],
    /// The proofs of leaf verifier or internal verifier in the execution order.
    pub proofs: Vec<Proof<SC>>,
}

impl<F: PrimeField32> InternalVmVerifierPvs<Felt<F>> {
    pub fn uninit<C: Config<F = F>>(builder: &mut Builder<C>) -> Self {
        Self {
            vm_verifier_pvs: VmVerifierPvs::<Felt<F>>::uninit(builder),
            leaf_verifier_commit: array::from_fn(|_| builder.uninit()),
            self_program_commit: array::from_fn(|_| builder.uninit()),
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
