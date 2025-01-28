use openvm_native_compiler::prelude::*;
use openvm_native_recursion::{hints::Hintable, vars::StarkProofVariable};
use openvm_stark_backend::config::StarkGenericConfig;
use openvm_stark_sdk::openvm_stark_backend::{config::Val, prover::types::Proof};

use super::types::MinimalVmVerifierInput;
use crate::{
    verifier::{leaf::types::UserPublicValuesRootProof, root::types::RootVmVerifierInput},
    C, SC,
};

#[derive(DslVariable, Clone)]
pub struct MinimalVmVerifierInputVariable<C: Config> {
    /// The proof of leaf verifier
    pub proof: StarkProofVariable<C>,
    /// Public values to expose
    pub public_values: Array<C, Felt<C::F>>,
}

impl<SC: StarkGenericConfig> MinimalVmVerifierInput<SC> {
    pub fn write_to_stream<C: Config<N = Val<SC>>>(&self) -> Vec<Vec<Val<SC>>>
    where
        Proof<SC>: Hintable<C>,
        UserPublicValuesRootProof<Val<SC>>: Hintable<C>,
    {
        let mut ret = Hintable::<C>::write(&self.proof);
        ret.extend(Hintable::<C>::write(&self.public_values_root_proof));
        ret
    }
}

impl Hintable<C> for MinimalVmVerifierInput<SC> {
    type HintVariable = MinimalVmVerifierInputVariable<C>;

    fn read(builder: &mut Builder<C>) -> Self::HintVariable {
        let proof = Proof::<SC>::read(builder);
        let public_values = Vec::<Val<SC>>::read(builder);
        Self::HintVariable {
            proof,
            public_values,
        }
    }

    fn write(&self) -> Vec<Vec<<C as Config>::N>> {
        let mut stream = self.proof.write();
        stream.extend(self.public_values_root_proof.write());
        stream
    }
}
