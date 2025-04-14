use std::array;

use openvm_native_compiler::prelude::*;
use openvm_native_recursion::hints::Hintable;
use openvm_stark_sdk::openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::FieldAlgebra,
    proof::Proof,
};

use crate::{
    verifier::{
        leaf::types::{LeafVmVerifierInput, UserPublicValuesRootProof},
        utils,
    },
    C, F,
};

#[derive(DslVariable, Clone)]
pub struct UserPublicValuesRootProofVariable<const CHUNK: usize, C: Config> {
    /// Sibling hashes for proving the merkle root of public values. For a specific VM, the path
    /// is constant. So we don't need the boolean which indicates if a node is a left child or
    /// right child.
    pub sibling_hashes: Array<C, [Felt<C::F>; CHUNK]>,
    pub public_values_commit: [Felt<C::F>; CHUNK],
}

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
        builder.range(0, len).for_each(|i_vec, builder| {
            let hash = array::from_fn(|_| builder.hint_felt());
            builder.set_value(&sibling_hashes, i_vec[0], hash);
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
        stream.extend(
            self.sibling_hashes
                .iter()
                .flat_map(utils::write_field_slice),
        );
        stream.extend(utils::write_field_slice(&self.public_values_commit));
        stream
    }
}
