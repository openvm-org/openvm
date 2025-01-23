use openvm_native_compiler::prelude::*;
use openvm_native_recursion::{hints::Hintable, vars::StarkProofVariable};
use openvm_stark_sdk::openvm_stark_backend::{config::Val, prover::types::Proof};

use super::types::MinimalVmVerifierInput;
use crate::{verifier::root::types::RootVmVerifierInput, C, SC};

#[derive(DslVariable, Clone)]
pub struct MinimalVmVerifierInputVariable<C: Config> {
    /// The proof of leaf verifier
    pub proof: StarkProofVariable<C>,
    /// Public values to expose
    pub public_values: Array<C, Felt<C::F>>,
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
        stream.extend(self.public_values.write());
        stream
    }
}
