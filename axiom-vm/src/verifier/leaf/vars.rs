use std::array;

use ax_stark_sdk::ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::AbstractField,
    prover::types::Proof,
};
use axvm_native_compiler::prelude::*;
use axvm_recursion::{
    hints::{Hintable, InnerVal},
    types::InnerConfig,
};

use crate::verifier::leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof};

#[derive(DslVariable, Clone)]
pub struct UserPublicValuesRootProofVariable<const CHUNK: usize, C: Config> {
    /// Sibling hashes for proving the merkle root of public values. For a specific VM, the path
    /// is constant. So we don't need the boolean which indicates if a node is a left child or right
    /// child.
    pub sibling_hashes: Array<C, [Felt<C::F>; CHUNK]>,
    pub public_values_commit: [Felt<C::F>; CHUNK],
}

type C = InnerConfig;
type F = InnerVal;

impl<SC: StarkGenericConfig> LeafVmVerifierInput<SC> {
    pub fn write_to_stream<C: Config<N = Val<SC>>>(&self) -> Vec<Vec<Val<SC>>>
    where
        Vec<Proof<SC>>: Hintable<C>,
        UserPublicValuesRootProof<Val<SC>>: Hintable<C>,
    {
        let mut ret = Hintable::<C>::write(&self.proofs);
        if let Some(pvs_root_proof) = &self.public_values_root_proof {
            ret.extend(Hintable::<C>::write(pvs_root_proof));
        }
        ret
    }
}

impl Hintable<C> for UserPublicValuesRootProof<F> {
    type HintVariable = UserPublicValuesRootProofVariable<{ DIGEST_SIZE }, C>;
    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let len = builder.hint_var();
        let sibling_hashes = builder.array(len);
        builder.range(0, len).for_each(|i, builder| {
            // FIXME: add hint support for slices.
            let hash = array::from_fn(|_| builder.hint_felt());
            builder.set_value(&sibling_hashes, i, hash);
        });
        let public_values_commit = array::from_fn(|_| builder.hint_felt());
        Self::HintVariable {
            sibling_hashes,
            public_values_commit,
        }
    }
    fn write(&self) -> Vec<Vec<<C as Config>::N>> {
        let len = <<C as Config>::N>::from_canonical_usize(self.sibling_hashes.len());
        let mut stream = len.write();
        stream.extend(self.sibling_hashes.iter().flat_map(write_field_slice));
        stream.extend(write_field_slice(&self.public_values_commit));
        stream
    }
}

fn write_field_slice(arr: &[F; DIGEST_SIZE]) -> Vec<Vec<<C as Config>::N>> {
    arr.iter().flat_map(|x| x.write()).collect()
}
