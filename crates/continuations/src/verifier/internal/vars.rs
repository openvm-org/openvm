use std::array;

use openvm_native_compiler::{ir::DIGEST_SIZE, prelude::*};
use openvm_native_recursion::{hints::Hintable, vars::StarkProofVariable};
use openvm_stark_sdk::openvm_stark_backend::proof::Proof;

use crate::{
    verifier::{
        internal::types::{InternalVmVerifierInput, VmStarkProof},
        utils::write_field_slice,
    },
    C, F, SC,
};

#[derive(DslVariable, Clone)]
pub struct InternalVmVerifierInputVariable<C: Config> {
    pub self_program_commit: [Felt<C::F>; DIGEST_SIZE],
    /// The proofs of the execution segments in the execution order.
    pub proofs: Array<C, StarkProofVariable<C>>,
}

#[derive(DslVariable, Clone)]
pub struct E2eStarkProofVariable<C: Config> {
    pub proof: StarkProofVariable<C>,
    pub user_public_values: Array<C, Felt<C::F>>,
}

impl Hintable<C> for InternalVmVerifierInput<SC> {
    type HintVariable = InternalVmVerifierInputVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let self_program_commit = array::from_fn(|_| builder.hint_felt());
        let proofs = Vec::<Proof<SC>>::read(builder);
        Self::HintVariable {
            self_program_commit,
            proofs,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::N>> {
        let mut stream = write_field_slice(&self.self_program_commit);
        stream.extend(self.proofs.write());
        stream
    }
}

impl Hintable<C> for VmStarkProof<SC> {
    type HintVariable = E2eStarkProofVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let proof = Proof::<SC>::read(builder);
        let user_public_values = Vec::<F>::read(builder);
        Self::HintVariable {
            proof,
            user_public_values,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::N>> {
        let mut stream = self.proof.write();
        stream.extend(self.user_public_values.write());
        stream
    }
}
